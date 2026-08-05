####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import datetime
from dateutil.relativedelta import relativedelta
from decimal import Decimal

def test_next_payment_date_standard_annual():
    assert _next_payment_date(datetime.date(2014, 1, 1), 1, None) == datetime.date(2015, 1, 1)

def test_next_payment_date_with_eom():
    assert _next_payment_date(datetime.date(2014, 1, 1), 1, 15) == datetime.date(2015, 1, 15)

def test_next_payment_date_quarterly():
    assert _next_payment_date(datetime.date(2023, 3, 10), 4, None) == datetime.date(2023, 6, 10)

def test_next_payment_date_monthly_with_invalid_eom():
    # Testing case where eom is 31 but next month has 30 days (April)
    assert _next_payment_date(datetime.date(2023, 3, 15), 12, 31) == datetime.date(2023, 4, 30)

def test_next_payment_date_decimal_frequency():
    assert _next_payment_date(datetime.date(2023, 1, 1), Decimal('2.0'), None) == datetime.date(2023, 7, 1)

def test_next_payment_date_leap_year_transition():
    assert _next_payment_date(datetime.date(2024, 2, 29), 1, None) == datetime.date(2025, 2, 28)
```


# LLM-generated content at query #2
#--------------------------

```python
def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}

def test_constructor_registry_property_is_empty_list():
    registry = DCCRegistryMachiniery()
    assert registry.registry == []

def test_constructor_table_property_is_empty_dict():
    registry = DCCRegistryMachinery()
    assert registry.table == {}
```


# LLM-generated content at query #3
#--------------------------

```python
def test_constructor_initializes_empty_buffers():
    registry_machinery = DCCRegistryMachinery()
    assert registry_machinery._buffer_main == {}
    assert registry_machinery._buffer_altn == {}

def test_constructor_initializes_new_instance_independently():
    registry_machinery_1 = DCCRegistryMachinery()
    registry_machinery_2 = DCCRegistryMachinery()
    assert registry_machinery_1 is not registry_machinery_2
    assert registry_machinery_1._buffer_main is not registry_machinery_2._buffer_main
    assert registry_machinery_1._buffer_altn is not registry_machinery << 0 >> registry_machinery_2._buffer_altn
```

Wait, I noticed a typo in my thought process for the last line. Correcting to:

```python
def test_constructor_initializes_empty_buffers():
    registry_machinery = DCCRegistryMachinery()
    assert registry_machinery._buffer_main == {}
    assert registry_machinery._buffer_altn == {}

def test_constructor_initializes_new_instance_independently():
    registry_machinery_1 = DCCRegistryMachinery()
    registry_machinery_2 = DCCRegistryMachinery()
    assert registry_machinery_1 is not registry_machinery_2
    assert registry_machinery_1._buffer_main is not registry_machinery_2._buffer_main
    assert registry_machinery_1._buffer_altn is not registry_machinery_2._buffer_altn
```


# LLM-generated content at query #4
#--------------------------

```python
def test_register_success():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="ACT/360", altnames={"Actual/360"}, currencies=set(), calculate_fraction_method=lambda x, y, z, f: 0)
    registry.register(dcc)
    assert registry.find("ACT/360") == dcc
    assert registry.find("Actual/360") == dcc
    assert dcc in registry.registry

def test_register_duplicate_main_name_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="ACT/360", altnames={"Actual/360"}, currencies=set(), calculate_fraction_method=lambda x, y, z, f: 0)
    dcc2 = DCC(name="ACT/360", altnames={"Other"}, currencies=set(), calculate_fraction_method=lambda x, y, z, f: 0)
    registry.register(dcc1)
    try:
        registry.register(dcc2)
    except TypeError as e:
        assert str(e) == "Day count convention 'ACT/360' is already registered"
    else:
        raise AssertionError("TypeError not raised")

def test_register_duplicate_altname_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="ACT/360", altnames={"Actual/360"}, currencies=set(), calculate_fraction_method=lambda x, y, z, f: 0)
    dcc2 = DCC(name="NEW", altnames={"Actual/360"}, currencies=set(), calculate_fraction_method=lambda x, y, z, f: 0)
    registry.register(dcc1)
    try:
        registry.register(dcc2)
    except TypeError as e:
        assert str(e) == "Day count convention 'NEW' is already registered"
    else:
        raise AssertionError("TypeError not raised")

def test_register_conflict_between_main_and_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="ACT/360", altnames={"Actual/360"}, currencies=set(), calculate_fraction_method=lambda x, y, z, f: 0)
    dcc2 = DCC(name="Actual/360", altnames={"Other"}, currencies=set(), calculate_fraction_method=lambda x, y, z, f: 0)
    registry.register(dcc1)
    try:
        registry.register(dcc2)
    except TypeError as e:
        assert str(e) == "Day count convention 'Actual/360' is already registered"
    else:
        raise AssertionError("TypeError not raised")
```


# LLM-generated content at query #5
#--------------------------

```python
def test_has_leap_day_contains_leap_day():
    from datetime import date
    import calendar
    from pypara.dcc import _has_leap_day
    start = date(2020, 1, 1)
    end = date(2020, 3, 1)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_boundary_leap_day():
    from datetime import date
    import calendar
    from pypara.dcc import _has_leap_day
    start = date(2020, 2, 29)
    end = date(2020, 2, 29)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_no_leap_day_in_range():
    from datetime import date
    import calendar
    from pypara.dcc import _has_leap_day
    start = date(2021, 1, 1)
    end = date(2021, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_range_around_leap_year_without_feb_29():
    from datetime import date
    import calendar
    from pypara.dcc import _has_leap_day
    start = date(2020, 3, 1)
    end = date(2021, 1, 1)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_long_range_with_leap_year():
    from datetime import date
    import calendar
    from pypara.dcc import _has_leap_day
    start = date(2019, 1, 1)
    end = date(2023, 1, 1)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_single_day_non_leap():
    from datetime import date
    import calendar
    from pypara.dcc import _has_leap_day
    start = date(2021, 6, 1)
    end = date(2021, 6, 1)
    assert _has_leap_day(start, end) == False
```


# LLM-generated content at query #6
#--------------------------

```python
def test_dcfc_30_360_isda_calculation_example_1():
    from datetime import date
    from decimal import Decimal
    ex1_start, ex1_asof = date(2007, 12, 28), date(2008, 2, 28)
    result = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_isda_calculation_example_2():
    from datetime import date
    from decimal import Decimal
    ex2_start, ex2_asof = date(2007, 12, 28), date(2008, 2, 29)
    result = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_isda_calculation_example_3():
    from datetime import date
    from decimal import Decimal
    ex3_start, ex3_asof = date(2007, 10, 31), date(2008, 11, 30)
    result = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_isda_calculation_example_4():
    from datetime import date
    from decimal import Decimal
    ex4_start, ex4_asof = date(2008, 2, 1), date(2009, 5, 31)
    result = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')

def test_dcfc_30_360_isda_start_day_31_adjustment():
    from datetime import date
    from decimal import Decimal
    # If start day is 31, it should be treated as 30.
    # (2023-01-31 to 2023-02-01) -> (2023-01-30 to 2023-02-01)
    # Days = (1 - 30) + 30*(2-1) + 360*(2023-2023) = -29 + 30 = 1 day.
    # Result = 1/360
    start = date(2023, 1, 31)
    asof = date(2023, 2, 1)
    end = date(2023, 2, 1)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    assert result == Decimal('1') / Decimal('360')

def test_dcfc_30_360_isda_both_days_31_adjustment():
    from datetime import date
    from decimal import Decimal
    # If start day is 30 and asof day is 31, asof becomes 30.
    # (2023-01-30 to 2023-01-31) -> (2023-01-30 to 2023-01-30) = 0 days.
    start = date(2023, 1, 30)
    asof = date(2023, 1, 31)
    end = date(2023, 1, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    assert result == Decimal('0')
```


# LLM-generated content at query #7
#--------------------------

```python
def test_calculate_fraction_valid_range():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple, Set, Optional
    
    class Currency: pass
    class DCFC:
        def __call__(self, start, asof, end, freq): return Decimal("0.5")
    
    DCC = NamedTuple("DCC", [("name", str), ("altnames", Set[str]), ("currencies", Set[Currency]), ("calculate_fraction_method", DCFC)])
    
    method = DCFC()
    dcc = DCC(name="Actual/360", altnames={"A/360"}, currencies=set(), calculate_fraction_method=method)
    
    start = date(2023, 1, 1)
    asof = date(2023, 1, 2)
    end = date(2023, 1, 3)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0.5")

def test_calculate_fraction_invalid_range_asof_too_early():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple, Set, Optional

    class Currency: pass
    class DCFC:
        def __call__(self, start, asof, end, freq): return Decimal("0.5")
    
    DCC = NamedTuple("DCC", [("name", str), ("altnames", Set[str]), ("currencies", Set[Currency]), ("calculate_fraction_method", DCFC)])
    ZERO = Decimal("0")
    
    method = DCFC()
    dcc = DCC(name="Actual/360", altnames=set(), currencies=set(), calculate_fraction_method=method)
    
    start = date(2023, 1, 5)
    asof = date(2023, 1, 2)
    end = date(2023, 1, 10)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == ZERO

def test_calculate_fraction_invalid_range_asof_too_late():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple, Set, Optional

    class Currency: pass
    class DCFC:
        def __call__(self, start, asof, end, freq): return Decimal("0.5")
    
    DCC = NamedTuple("DCC", [("name", str), ("altnames", Set[str]), ("currencies", Set[Currency]), ("calculate_fraction_method", DCFC)])
    ZERO = Decimal("0")
    
    method = DCFC()
    dcc = DCC(name="Actual/360", altnames=set(), currencies=set(), calculate_fraction_method=method)
    
    start = date(2023, 1, 1)
    asof = date(2023, 1, 15)
    end = date(2023, 1, 10)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == ZERO

def test_calculate_fraction_with_frequency():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple, Set, Optional

    class Currency: pass
    class DCFC:
        def __call__(self, start, asof, end, freq): return freq if freq is not None else Decimal("0")
    
    DCC = NamedTuple("DCC", [("name", str), ("altnames", Set[str]), ("currencies", Set[Currency]), ("calculate_fraction_method", DCFC)])
    
    method = DCFC()
    dcc = DCC(name="Actual/360", altnames=set(), currencies=set(), calculate_fraction_method=method)
    
    start = date(2020, 1, 1)
    asof = date(2020, 1, 2)
    end = date(2020, 1, 3)
    freq = Decimal("2.0")
    
    result = dcc.calculate_fraction(start, asof, end, freq=freq)
    assert result == Decimal("2.0")
```


# LLM-generated content at query #8
#--------------------------

```python
def test_register_raises_type_error_on_duplicate_altname():
    registry = DCCRegistryMachinery()
    method = lambda s, a, e, f: Decimal('0.1')
    dcc1 = DCC(name="ACT/360", altnames={"ACT360", "ACT/360_ALT"}, currencies=set(), calculate_fraction_method=method)
    dcc2 = DCC(name="30/360", altnames={"ACT360"}, currencies=set(), calculate_fraction_method=method)
    registry.register(dcc1)
    try:
        registry.register(dcc2)
    except TypeError as e:
        assert str(e) == "Day count convention '30/360' is already registered"
        return
    raise AssertionError("TypeError was not raised for duplicate altname")

```


# LLM-generated content at query #9
#--------------------------

```python
def test_find_returns_none_when_empty():
    registry = DCCRegistryMachinery()
    assert registry.find("NonExistent") is None

def test_find_returns_dcc_by_main_name():
    class MockDCC:
        def __init__(self, name, altnames):
            self.name = name
            self.altnames = altnames
    
    dcc = MockDCC("ACT/360", ["Act/360"])
    registry = DCCRegistryMachinery()
    registry.register(dcc)
    assert registry.find("ACT/360") is dcc

def test_find_returns_dcc_by_altname():
    class MockDCC:
        def __init__(self, name, altnames):
            self.name = name
            self.altnames = altnames
            
    dcc = MockDCC("ACT/360", ["ACT/360", "Actual/360"])
    registry = DCCRegistryMachinery()
    registry.register(dcc)
    assert registry.find("Actual/360") is dcc

def test_find_handles_case_insensitivity_and_stripping():
    class MockDCC:
        def __init__(self, name, altnames):
            self.name = name
            self.altnames = altnames
            
    dcc = MockDCC("ACT/360", ["ACT/360"])
    registry = DCCRegistryMachinery()
    registry.register(dcc)
    assert registry.find("  act/360  ") is dcc
    assert registry.find("act/360") is dcc

def test_find_prefers_exact_match_over_normalized():
    class MockDCC:
        def __init__(self, name, altnames):
            self.name = name
            self.altnames = altnames
            
    dcc1 = MockDCC("FIRST", [])
    dcc2 = MockDCC("SECOND", [])
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    # Manually inject into buffer_main to bypass register collision check for testing find logic
    registry._buffer_main["SECOND"] = dcc2 
    
    assert registry.find("FIRST") is dcc1
```


# LLM-generated content at query #10
#--------------------------

```python
import datetime
import calendar

def test_is_last_day_of_month_true_january():
    date = datetime.date(2023, 1, 31)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_false_january():
    date = datetime.date(2023, 1, 30)
    assert _is_last_day_of_month(date) == False

def test_is_last_day_of_month_true_leap_year_february():
    date = datetime.date(2024, 2, 29)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_false_leap_year_february():
    date = datetime.date(2024, 2, 28)
    assert _is_last_day_of_month(date) == False

def test_is_last_day_of_month_true_non_leap_year_february():
    date = datetime.date(2023, 2, 28)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_true_april_30():
    date = datetime.date(2023, 4, 30)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_false_april_29():
    date = datetime.date(2023, 4, 29)
    assert _is_last_day_of_month(date) == False
```


# LLM-generated content at query #11
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_act_act_icma_calculation():
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal("2")
    # p1 = (2019-09-10 - 2019-03-02) = 191 days
    # p2 = (2020-03-02 - 2019-03-02) = 366 days (leap year)
    # result = 191 / 366 / 2 = 0.260928961748...
    expected = Decimal("191") / Decimal("366") / Decimal("2")
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert round(result, 10) == round(expected, 10)

def test_dcfc_act_act_icma_default_freq():
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    # freq defaults to ONE (assumed from code context)
    expected = Decimal("191") / Decimal("366") / Decimal("1")
    result = dcfc_act_act_icma(start, asof, end)
    assert round(result, 10) == round(expected, 10)

def test_dcfc_act_act_icma_invalid_date_range():
    # If not start <= asof <= end, it should return ZERO if we were testing DCC.calculate_fraction
    # But dcfc_act_act_icma doesn't have the check internally, its parent registry does.
    # However, let's test the math when asof > end is passed directly to this function.
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2021, 3, 2)
    end = datetime.date(2020, 3, 2)
    # p1 = 731 days, p2 = 366 days, freq=1 -> 731/366
    expected = Decimal("731") / Decimal("366")
    result = dcfc_act_act_icma(start, asof, end)
    assert result == expected

def test_dcfc_act_act_icma_same_dates():
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 1, 1)
    end = datetime.date(2019, 1, 1)
    # p1=0, p2=0 -> This would raise DivisionByZero in the implementation provided.
    # We test that it behaves according to the math logic (p1/p2).
    # Since we cannot use try/except or if, we only test valid non-zero p2.
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 1, 1)
    end = datetime.date(2019, 1, 2)
    expected = Decimal("0") / Decimal("1") / Decimal("1")
    result = dcfc_act_act_icma(start, asof, end)
    assert result == Decimal("0")
```


# LLM-generated content at query #12
#--------------------------

```python
def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}

def test_constructor_registry_property_is_empty():
    registry = DCCRegistryMachiniery()
    assert registry.registry == []

def test_constructor_table_property_is_empty():
    registry = DCCRegistryMachinery()
    assert registry.table == {}
```


# LLM-generated content at query #13
#--------------------------

```python
def test_interest_valid_dates():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple, Set, Optional, Union
    
    class Currency: pass
    class Money(Decimal): pass
    class Date: 
        def __init__(self, d): self.d = d
        def __le__(self, other): return self.d <= other.d
        def __lt__(self, other): return self.d < other.d
        def __ge__(self, other): return self.d >= other.d
    
    class DCC(NamedTuple):
        name: str
        altnames: Set[str]
        currencies: Set[Currency]
        calculate_fraction_method: callable
        def calculate_fraction(self, start, asof, end, freq=None):
            if not start <= asof <= end: return Decimal('0')
            return self[3](start, asof, end, freq)
        def interest(self, principal, rate, start, asof, end=None, freq=None):
            return principal * rate * self.calculate_fraction(start, asof, end or asof, freq)

    mock_method = lambda s, a, e, f: Decimal('0.5')
    dcc_instance = DCC("Test", set(), set(), mock_method)
    
    principal = Money(Decimal('1000'))
    rate = Decimal('0.05')
    start_date = Date(date(2023, 1, 1))
    asof_date = Date(date(2023, 6, 1))
    end_date = Date(date(2023, 12, 31))
    
    expected_interest = Decimal('25.0') # 1000 * 0.05 * 0.5
    actual_interest = dcc_instance.interest(principal, rate, start_date, asof_date, end_date)
    
    assert actual_interest == expected_interest

def test_interest_invalid_dates_returns_zero():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple, Set, Optional, Union

    class Currency: pass
    class Money(Decimal): pass
    class Date: 
        def __init__(self, d): self.d = d
        def __le__(self, other): return self.d <= other.d
        def __lt__(self, other): return self.d < other.d
        def __ge__(self, other): return self.d >= other.d

    class DCC(NamedTuple):
        name: str
        altnames: Set[str]
        currencies: Set[Currency]
        calculate_fraction_method: callable
        def calculate_fraction(self, start, asof, end, freq=None):
            if not start <= asof <= end: return Decimal('0')
            return self[3](start, asof, end, freq)
        def interest(self, principal, rate, start, asof, end=None, freq=None):
            return principal * rate * self.calculate_fraction(start, asof, end or asof, freq)

    mock_method = lambda s, a, e, f: Decimal('0.5')
    dcc_instance = DCC("Test", set(), set(), mock_method)
    
    principal = Money(Decimal('1000'))
    rate = Decimal('0.05')
    start_date = Date(date(2023, 6, 1))
    asof_date = Date(date(2023, 1, 1)) # asof < start
    end_date = Date(date(2023, 12, 31))
    
    actual_interest = dcc_instance.interest(principal, rate, start_date, asof_date, end_date)
    
    assert actual_interest == Decimal('0')

def test_interest_end_date_is_asof_when_none():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple, Set, Optional, Union

    class Currency: pass
    class Money(Decimal): pass
    class Date: 
        def __init__(self, d): self.d = d
        def __le__(self, other): return self.d <= other.d
        def __lt__(self, other): return self.d < other.d
        def __ge__(self, other): return self.d >= other.d

    class DCC(NamedTuple):
        name: str
        altnames: Set[str]
        currencies: Set[Currency]
        calculate_fraction_method: callable
        def calculate_fraction(self, start, asof, end, freq=None):
            if not start <= asof <= end: return Decimal('0')
            return self[3](start, asof, end, freq)
        def interest(self, principal, rate, start, asof, end=None, freq=None):
            return principal * rate * self.calculate_fraction(start, asof, end or asof, freq)

    mock_method = lambda s, a, e, f: Decimal('1.0')
    dcc_instance = DCC("Test", set(), set(), mock_method)
    
    principal = Money(Decimal('100'))
    rate = Decimal('0.1')
    start_date = Date(date(2023, 1, 1))
    asof_date = Date(date(2023, 1, 1))
    # end is None, so it should use asof (which is start/asof) -> fraction 1.0
    
    actual_interest = dcc_instance.interest(principal, rate, start_date, asof_date, end=None)
    
    assert actual_interest == Decimal('10.0')
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
from decimal import Decimal
import datetime

def test_dcfc_30_e_360_standard_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    assert result == Decimal('0.1666666666666666666666666667')

def test_dcfc_30_e_360_leap_year_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    assert result == Decimal('0.1694444444444444444444444444')

def test_dcfc_30_e_360_end_of_month_case():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    assert result == Decimal('1.0833333333333333333333333333')

def test_dcfc_30_e_360_long_period_case():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    assert result == Decimal('1.3305555555555555555555555556')

def test_dcfc_30_e_360_day_31_adjustment_start():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    # 31st becomes 30th -> (28-30) + 30*(2-1) + 360*(2023-2023) = -2 + 30 = 28 days. 28/360
    assert result == Decimal('28') / Decimal('360')

def test_dcfc_30_e_360_day_31_adjustment_asof():
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 3, 31)
    end = datetime.date(2023, 3, 31)
    result = dcf_30_e_360(start=start, asof=asof, end=end)
    # 31st becomes 30th -> (30-1) + 30*(3-1) + 0 = 29 + 60 = 89 days. 89/360
    assert result == Decimal('89') / Decimal('360')
```


# LLM-generated content at query #2
#--------------------------

```python
def test_register_success():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/360", altnames={"ACT/360", "Actual/360"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: 0.1)
    registry.register(dcc)
    assert registry.find("Act/360") == dcc
    assert registry.find("ACT/360") == dcc
    assert registry.find("Actual/360") == dcc
    assert len(registry.registry) == 1

def test_register_duplicate_main_name_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Act/360", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: 0.1)
    dcc2 = DCC(name="Act/360", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: 0.2)
    registry.register(dcc1)
    try:
        registry.register(dcc2)
    except TypeError as e:
        assert str(e) == "Day count convention 'Act/360' is already registered"

def test_register_duplicate_altname_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Act/360", altnames={"ACT/360"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: 0.1)
    dcc2 = DCC(name="Other", altnames={"ACT/360"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: 0.2)
    registry.register(dcc1)
    try:
        registry.register(dcc2)
    except TypeError as e:
        assert str(e) == "Day count convention 'Other' is already registered"

def test_register_conflict_within_same_dcc_altnames():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/360", altnames={"Act/360", "Duplicate"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: 0.1)
    # Note: The implementation checks if name is registered, then iterates altnames. 
    # If an altname equals the main name, it might trigger depending on order.
    # However, looking at the code: it checks self._is_registered(dcc.name) first.
    # If we provide a dcc where altnames contains its own name, the loop will check if name is registered (True).
    try:
        registry.register(dcc)
    except TypeError as e:
        assert "already registered" in str(e)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_construct_date_valid():
    from datetime import date
    from pypara.dcc import _construct_date
    assert _construct_date(2023, 10, 5) == date(2023, 10, 5)

def test_construct_date_invalid_year():
    from pypara.dcc import _construct_date
    try:
        _construct_date(0, 1, 1)
    except ValueError as e:
        assert str(e) == "year, month and day must be greater than 0."

def test_construct_date_invalid_month():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, -1, 1)
    except ValueError as e:
        assert str(e) == "year, month and day must be greater than 0."

def test_construct_date_invalid_day():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, 1, -5)
    except ValueError as e:
        assert str(e) == "year, month and day must be greater than 0."

def test_construct_date_auto_decrement_day():
    from datetime import date
    from pypara.dcc import _construct_date
    # Attempt to create Feb 30th, should decrement day until it hits valid Feb 28/29
    # Depending on year provided (non-leap), it will recursively find the last valid day
    result = _construct_date(2023, 2, 30)
    assert result == date(2023, 2, 28)

def test_construct_date_auto_decrement_leap_year():
    from datetime import date
    from pypara.dcc import _construct_date
    # Attempt to create Feb 30th in a leap year
    result = _construct_date(2024, 2, 30)
    assert result == date(2024, 2, 29)

def test_construct_date_invalid_month_range():
    from pypara.dcc import _construct_date
    # Month 13 is not handled by the 'day out of range' logic, so it should raise ValueError from datetime.date
    try:
        _construct_date(2023, 13, 1)
    except ValueError as e:
        assert str(e) == "month must be in 1..12"
```


# LLM-generated content at query #4
#--------------------------

```python
import datetime
from decimal import Decimal
from pypara.dcc import dcfc_30_e_plus_360

def test_dcfc_30_e_plus_360_case1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    result = dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')

def test_dcfc_30_e_plus_360_case2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    result = dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')

def test_dcfc_30_e_plus_360_case3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    result = dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')

def test_dcfc_30_e_plus_360_case4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    result = dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')

def test_dcfc_30_e_plus_360_boundary_start_31():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    # start becomes 2023-01-30. nod = (28-30) + 30*(2-1) + 360*(0) = -2 + 30 = 28. 28/360 = 0.0777...
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=end)
    assert result == Decimal('28') / Decimal('360')

def test_dcfc_30_e_plus_360_boundary_asof_31():
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 31)
    end = datetime.date(2023, 1, 31)
    # asof becomes 2023-02-01. nod = (1-1) + 30*(2-1) + 360*(0) = 30. 30/360 = 0.0833...
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=end)
    assert result == Decimal('30') / Decimal('360')
```


# LLM-generated content at query #5
#--------------------------

```python
def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}

def test_constructor_registry_property_is_empty_list():
    registry = DCCRegistryMachiniery()
    assert registry.registry == []

def test_constructor_table_property_is_empty_dict():
    registry = DCCRegistryMachinery()
    assert registry.table == {}
```


# LLM-generated content at query #6
#--------------------------

```python
from decimal import Decimal
import datetime

def test_dcfc_30_360_isda_standard_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    assert result == Decimal('0.1666666666666666666666666667')

def test_dcfc_30_360_isda_leap_year_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    assert result == Decimal('0.1694444444444444444444444444')

def test_dcfc_30_360_isda_month_end_case():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    assert result == Decimal('1.0833333333333333333333333333')

def test_dcfc_30_360_isda_long_period_case():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    assert result == Decimal('1.3333333333333333333333333333')

def test_dcfc_30_360_isda_day_31_adjustment():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    assert result == Decimal('27/360')

def test_dcfc_30_360_isda_both_days_31_adjustment():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    end = datetime.date(2023, 3, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    assert result == Decimal('60/360')
```


# LLM-generated content at query #7
#--------------------------

```python
def test_register_raises_type_error_on_duplicate_main_name():
    machinery = DCCRegistryMachinery()
    dcc1 = DCC(name="ACT/360", altnames={"ACT360"}, currencies=set(), calculate_fraction_method=lambda x, y, z, f: 0)
    dcc2 = DCC(name="ACT/360", altnames={"OTHER"}, currencies=set(), calculate_fraction_method=lambda x, y, z, f: 0)
    machinery.register(dcc1)
    try:
        machinery.register(dcc2)
        raise AssertionError("TypeError was not raised")
    except TypeError as e:
        assert str(e) == "Day count convention 'ACT/360' is already registered"
```


# LLM-generated content at query #8
#--------------------------

```python
def test_interest_valid_dates():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple, Set, Optional, Union
    
    class Currency: pass
    class Money(Decimal): pass
    class Date: 
        def __init__(self, d): self.d = d
        def __le__(self, other): return self.d <= other.d
        def __lt__(self, other): return self.d < other.d
        def __ge__(self, other): return self.d >= other.d
        def __gt__(self, other): return self.d > other.d

    class DCC(NamedTuple):
        name: str
        altnames: Set[str]
        currencies: Set[Currency]
        calculate_fraction_method: callable
        def calculate_fraction(self, start, asof, end, freq=None):
            if not start <= asof <= end: return Decimal('0')
            return self[3](start, asof, end, freq)
        def interest(self, principal, rate, start, asof, end=None, freq=None):
            return principal * rate * self.calculate_fraction(start, asof, end or asof, freq)

    def mock_method(s, a, e, f): return Decimal('0.5')
    
    dcc = DCC("Test", set(), set(), mock_method)
    principal = Money(Decimal('1000'))
    rate = Decimal('0.05')
    start = Date(date(2023, 1, 1))
    asof = Date(date(2023, 6, 1))
    end = Date(date(2023, 12, 31))
    
    expected_interest = Decimal('1000') * Decimal('0.05') * Decimal('0.5')
    actual_interest = dcc.interest(principal, rate, start, asof, end)
    
    assert actual_interest == expected_interest

def test_interest_invalid_dates_returns_zero():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple, Set, Optional, Union

    class Currency: pass
    class Money(Decimal): pass
    class Date: 
        def __init__(self, d): self.d = d
        def __le__(self, other): return self.d <= other.d
        def __lt__(self, other): return self.d < other.d
        def __ge__(self, other): return self.d >= other.d
        def __gt__(self, other): return self.d > other.d

    class DCC(NamedTuple):
        name: str
        altnames: Set[str]
        currencies: Set[Currency]
        calculate_fraction_method: callable
        def calculate_fraction(self, start, asof, end, freq=None):
            if not start <= asof <= end: return Decimal('0')
            return self[3](start, asof, end, freq)
        def interest(self, principal, rate, start, asof, end=None, freq=None):
            return principal * rate * self.calculate_fraction(start, asof, end or asof, freq)

    def mock_method(s, a, e, f): return Decimal('0.5')
    
    dcc = DCC("Test", set(), set(), mock_method)
    principal = Money(Decimal('1000'))
    rate = Decimal('0.05')
    start = Date(date(2023, 6, 1))
    asof = Date(date(2023, 1, 1)) # asof < start
    end = Date(date(2023, 12, 31))
    
    actual_interest = dcc.interest(principal, rate, start, asof, end)
    assert actual_interest == Decimal('0')

def test_interest_default_end_date():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple, Set, Optional, Union

    class Currency: pass
    class Money(Decimal): pass
    class Date: 
        def __init__(self, d): self.d = d
        def __le__(self, other): return self.d <= other.d
        def __lt__(self, other): return self.d < other.d
        def __ge__(self, other): return self.d >= other.d
        def __gt__(self, other): return self.d > other.d

    class DCC(NamedTuple):
        name: str
        altnames: Set[str]
        currencies: Set[Currency]
        calculate_fraction_method: callable
        def calculate_fraction(self, start, asof, end, freq=None):
            if not start <= asof <= end: return Decimal('0')
            return self[3](start, asof, end, freq)
        def interest(self, principal, rate, start, asof, end=None, freq=None):
            return principal * rate * self.calculate_fraction(start, asof, end or asof, freq)

    def mock_method(s, a, e, f): return Decimal('1.0')
    
    dcc = DCC("Test", set(), set(), mock_method)
    principal = Money(Decimal('1000'))
    rate = Decimal('0.10')
    start = Date(date(2023, 1, 1))
    asof = Date(date(2023, 2, 1))
    # end is None, so end should become asof (2023-02-01)
    
    expected_interest = Decimal('1000') * Decimal('0.10') * Decimal('1.0')
    actual_interest = dcc.interest(principal, rate, start, asof)
    
    assert actual_interest == expected_interest
```


# LLM-generated content at query #9
#--------------------------

```python
from decimal import Decimal
import datetime

def test_dcfc_30_360_us_standard_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_us_leap_year_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_us_month_end_case():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_us_long_period_case():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('1.33333333333333')

def test_dcfc_30_360_us_zero_days():
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 1)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    assert result == Decimal('0')
```


# LLM-generated content at query #10
#--------------------------

```python
import datetime
import calendar
from pypara.dcc import _has_leap_day

def test_has_leap_day_true_within_range():
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 3, 1)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_true_boundary_start():
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 3, 1)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_true_boundary_end():
    start = datetime.date(2020, 2, 28)
    end = datetime.date(2020, 2, 29)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_false_no_leap_year_in_range():
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_false_leap_year_exists_but_february_not_in_range():
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_false_leap_year_exists_but_after_range():
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2024, 1, 1)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_true_spanning_multiple_years_including_leap":
    start = datetime.date(2019, 1, 1)
    end = datetime.date(2021, 1, 1)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_false_non_leap_century():
    start = datetime.date(1900, 1, 1)
    end = datetime.date(1900, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_true_leap_century():
    start = datetime.date(2000, 1, 1)
    end = datetime.date(2000, 12, 31)
    assert _has_leap_day(start, end) == True
```


# LLM-generated content at query #11
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_30_360_german_standard_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_german_leap_year_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_german_month_end_case():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcf_30_360_german(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_german_long_period_case():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('1.33055555555556')

def test_dcfc_30_360_german_day_31_adjustment():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # d1 becomes 30, d2 is 28. (28-30) + 30*(2-1) + 360*(0) = -2 + 30 = 28. 28/360
    assert result == Decimal('28') / Decimal('360')

def test_dcfc_30_360_german_february_leap_adjustment():
    start = datetime.date(2024, 2, 29)
    asof = datetime.date(2024, 3, 1)
    end = datetime.date(2024, 3, 1)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # start is last day of Feb in leap year -> d1 = 30. asof is 1 -> d2 = 1.
    # (1-30) + 30*(3-2) + 0 = -29 + 30 = 1. Result 1/360
    assert result == Decimal('1') / Decimal('360')
```


# LLM-generated content at query #12
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_30_360_german_predicate_true_via_day_31():
    start_date = datetime.date(2023, 1, 31)
    asof_date = datetime.date(2023, 2, 1)
    end_date = datetime.date(2023, 2, 1)
    # The predicate is: start.day == 31 or (start.month == 2 and _is_last_day_of_month(start))
    # Since start.day is 31, the first part of 'or' is True.
    from pypara.dcc import dcfc_30_360_german
    result = dcfc_30_360_german(start=start_date, asof=asof_date, end=end_date)
    assert result == Decimal('0') # (30-30 + 30*(2-1) + 360*(2023-2023)) / 360 = 30/360 is not the logic here, but we just need to trigger line 25.
    # Actually, let's verify the predicate manually in the test via the function call logic.
    # The instruction asks to ensure the predicate at line 25 evaluates to True.

def test_dcfc_30_360_german_predicate_true_via_february_last_day():
    start_date = datetime.date(2024, 2, 29) # Leap year last day
    asof_date = datetime.date(2024, 3, 1)
    end_date = datetime.date(2024, 3, 1)
    from pypara.dcc import dcfc_30_360_german
    # For Feb 29, 2024: start.month == 2 and _is_last_day_of_month(start) is True.
    result = dcfv_30_360_german_helper(start_date, asof_date, end_date)
    assert result is not None

def test_dcfc_30_360_german_predicate_true_via_february_last_day_non_leap():
    start_date = datetime.date(2023, 2, 28) # Non-leap year last day
    asof_date = datetime.date(2023, 3, 1)
    end_date = datetime.date(2023, 3, 1)
    from pypara.dcc import dcfc_30_360_german
    # For Feb 28, 2023: start.month == 2 and _is_last_day_of_month(start) is True.
    result = dcfc_30_360_german(start=start_date, asof=asof_date, end=end_date)
    assert result is not None

def dcfv_30_360_german_helper(start, asof, end):
    from pypara.dcc import dcfc_30_360_german
    return dcfc_30_360_german(start=start, asof=asof, end=end)

# Since I cannot define functions, I will provide a single clean test case that triggers the logic.
```

Wait, I must follow the rules: No custom function definitions or control structures in the response. Only variable assignments, assertions and calls.

```python
def test_dcfc_30_360_german_predicate_true_day_31():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    start = date(2023, 5, 31)
    asof = date(2023, 6, 1)
    end = date(2023, 6, 1)
    # This triggers: start.day == 31 -> True
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    assert result is not None

def test_dcfc_30_360_german_predicate_true_february_end():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    start = date(2024, 2, 29)
    asof = date(2024, 3, 1)
    end = date(2024, 3, 1)
    # This triggers: (start.month == 2 and _is_last_day_of_month(start)) -> True
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    assert result is not None
```


# LLM-generated content at query #13
#--------------------------

```python
def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}

def test_constructor_property_registry_is_empty():
    registry = DCCRegistryMachiniery()
    assert registry.registry == []

def test_constructor_property_table_is_empty():
    registry = DCCRegistryMachinery()
    assert registry.table == {}
```


# LLM-generated content at query #14
#--------------------------

```python
def test_last_payment_date_basic_annual():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(2014, 1, 1), date(2015, 12, 31), 1) == date(2015, 1, 1)

def test_last_payment_date_annual_same_year():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(2015, 1, 1), date(2015, 12, 31), 1) == date(2015, 1, 1)

def test_last_payment_date_semi_annual():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(201mu, 1, 1), date(2015, 12, 31), 2) == date(2015, 7, 1)

def test_last_payment_date_semi_annual_mid_year():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(2014, 1, 1), date(2015, 8, 31), 2) == date(2015, 7, 1)

def test_last_payment_date_semi_annual_early_year():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(2014, 1, 1), date(2015, 4, 30), 2) == date(2015, 1, 1)

def test_last_payment_date_asof_before_start():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(2014, 6, 1), date(2015, 4, 30), 1) == date(2014, 6, 1)

def test_last_payment_date_quarterly():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(2008, 7, 7), date(2015, 10, 6), 4) == date(2015, 7, 7)

def test_last_payment_date_annual_late_year():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(2014, 12, 9), date(2015, 12, 4), 1) == date(2014, 12, 9)

def test_last_payment_date_semi_annual_long_range():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(2012, 12, 15), date(2016, 1, 6), 2) == date(2015, 12, 15)

def test_last_payment_date_semi_annual_short_range():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(2012, 12, 15), date(2015, 12, 31), 2) == date(2015, 12, 15)

def test_last_payment_date_with_eom_parameter():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(2014, 1, 1), date(2015, 12, 31), 1, eom=28) == date(2015, 1, 28)

def test_last_payment_date_eom_boundary_condition():
    from datetime import date
    from pypara.dcc import _last_payment_date
    assert _last_payment_date(date(2014, 1, 31), date(2015, 1, 30), 1, eom=31) == date(2014, 1, 31)
```


# LLM-generated content at query #15
#--------------------------

def test_dcfc_nl_365_standard_calculation():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_nl_365(start, asof, end)
    assert round(result, 14) == Decimal('0.16986301369863')

def test_dcfc_nl_365_with_leap_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_nl_365(start, asof, end)
    assert round(result, 14) == Decimal('0.16986301369863')

def test_dcfc_nl_365_long_period():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_nl_365(start, asof, end)
    assert round(result, 14) == Decimal('1.08219178082192')

def test_dcfc_nl_365_extended_leap_year_range():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_nl_365(start, asof, end)
    assert round(result, 14) == Decimal('1.32602739726027')

def test_dcfc_nl_365_same_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 1)
    result = dcfc_nl_365(start, asof, end)
    assert result == Decimal('0')


# LLM-generated content at query #16
#--------------------------

```python
def test_find_returns_correct_dcc_when_exact_name_exists():
    registry = DCCRegistryMachinery()
    class MockDCC:
        def __init__(self, name):
            self.name = name
            self.altnames = []
    mock_dcc = MockDCC("Act/360")
    registry.register(mock_dcc)
    assert registry.find("Act/360") == mock_dcc

def test_find_returns_correct_dcc_when_altname_exists():
    registry = DCCRegistryMachinery()
    class MockDCC:
        def __init__(self, name, altnames):
            self.name = name
            self.altnames = altnames
    mock_dcc = MockDCC("Act/360", ["ACTUAL/360"])
    registry.register(mock_dcc)
    assert registry.find("ACTUAL/360") == mock_dcc

def test_find_returns_correct_dcc_with_case_insensitivity_and_stripping():
    registry = DCCRegistryMachinery()
    class MockDCC:
        def __init__(self, name):
            self.name = name
            self.altnames = []
    mock_dcc = MockDCC("ACT/360")
    registry.register(mock_dcc)
    assert registry.find("  act/360  ") == mock_dcc

def test_find_returns_none_when_not_found():
    registry = DCCRegistryMachinery()
    class MockDCC:
        def __init__(self, name):
            self.name = name
            self.altnames = []
    mock_dcc = MockDCC("Act/360")
    registry.register(mock_dcc)
    assert registry.find("NonExistent") is None

def test_find_returns_correct_dcc_for_alternative_name_with_padding():
    registry = DCCRegistryMachinery()
    class MockDCC:
        def __init__(self, name, altnames):
            self.name = name
            self.altnames = altnames
    mock_dcc = MockDCC("30/360 US", ["30/360 US ISDA"])
    registry.register(mock_dcc)
    assert registry.find("  30/360 US ISDA  ") == mock_dcc
```


# LLM-generated content at query #17
#--------------------------

```python
import datetime

def test_get_date_range_basic():
    from pypara.dcc import _get_date_range
    class MockDate:
        def __init__(self, day):
            self.day = day
        def __sub__(self, other):
            return datetime.timedelta(days=self.day - other.day)
        def __add__(self, delta):
            return MockDate(self.day + delta.days)
        def __eq__(self, other):
            return self.day == other.date_val
        @property
        def date_val(self):
            return self.day

    start = MockDate(1)
    end = MockDate(4)
    # end is exclusive, so range should be days 1, 2, 3 (delta = 3)
    result = list(_get_date_range(start, end))
    assert len(result) == 3
    assert result[0].day == 1
    assert result[1].day == 2
    assert result[2].day == 3

def test_get_date_range_empty():
    from pypara.dcc import _get_date_range
    class MockDate:
        def __init__(self, day):
            self.day = day
        def __sub__(self, other):
            return datetime.timedelta(days=self.day - other.days)
        def __add__(self, delta):
            return MockDate(self.day + delta.days)

    start = MockDate(5)
    end = MockDate(5)
    result = list(_get_date_range(start, end))
    assert len(result) == 0

def test_get_date_range_negative_delta():
    from pypara.dcc import _get_date_range
    class MockDate:
        def __init__(self, day):
            self.day = day
        def __sub__(self, other):
            return datetime.timedelta(days=self.day - other.days)
        def __add__(self, delta):
            return MockDate(self.day + delta.days)

    start = MockDate(10)
    end = MockDate(5)
    result = list(_get(start, end))
    # range((5-10)) -> range(-5) is empty
    assert len(result) == 0
```


