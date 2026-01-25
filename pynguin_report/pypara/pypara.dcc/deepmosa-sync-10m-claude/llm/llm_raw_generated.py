####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_last_day_of_month_last_day():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date: date) -> bool:
        return date.day == calendar.monthrange(date.year, date.month)[1]
    
    result = _is_last_day_of_month(date(2023, 1, 31))
    assert result is True


def test_is_last_day_of_month_not_last_day():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date: date) -> bool:
        return date.day == calendar.monthrange(date.year, date.month)[1]
    
    result = _is_last_day_of_month(date(2023, 1, 30))
    assert result is False


def test_is_last_day_of_month_february_leap_year():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date: date) -> bool:
        return date.day == calendar.monthrange(date.year, date.month)[1]
    
    result = _is_last_day_of_month(date(2020, 2, 29))
    assert result is True


def test_is_last_day_of_month_february_non_leap_year():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date: date) -> bool:
        return date.day == calendar.monthrange(date.year, date.month)[1]
    
    result = _is_last_day_of_month(date(2023, 2, 28))
    assert result is True


def test_is_last_day_of_month_february_non_leap_year_not_last():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date: date) -> bool:
        return date.day == calendar.monthrange(date.year, date.month)[1]
    
    result = _is_last_day_of_month(date(2023, 2, 27))
    assert result is False


def test_is_last_day_of_month_april_30_days():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date: date) -> bool:
        return date.day == calendar.monthrange(date.year, date.month)[1]
    
    result = _is_last_day_of_month(date(2023, 4, 30))
    assert result is True


def test_is_last_day_of_month_december_last_day():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date: date) -> bool:
        return date.day == calendar.monthrange(date.year, date.month)[1]
    
    result = _is_last_day_of_month(date(2023, 12, 31))
    assert result is True


# LLM-generated content at query #2
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
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    
    result = dcfc_act_act_icma(start, asof, end)
    assert result == Decimal('0')


def test_dcfc_act_act_icma_with_freq():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 1)
    freq = Decimal('2')
    
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert result > Decimal('0')


def test_dcfc_act_act_icma_full_period():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 1, 1)
    
    result = dcfc_act_act_icma(start, asof, end)
    assert result == Decimal('1')


def test_dcfc_act_act_icma_one_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    end = datetime.date(2020, 1, 366)
    
    result = dcfc_act_act_icma(start, asof, end)
    assert result > Decimal('0')


def test_dcfc_act_act_icma_half_year():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2021, 1, 1)
    
    result = dcfc_act_act_icma(start, asof, end)
    assert Decimal('0.49') < result < Decimal('0.51')


def test_dcfc_act_act_icma_with_none_freq():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 1)
    
    result_with_none = dcfc_act_act_icma(start, asof, end, None)
    result_with_one = dcfc_act_act_icma(start, asof, end, Decimal('1'))
    
    assert result_with_none == result_with_one


# LLM-generated content at query #3
#--------------------------

```python
def test_register_success():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames={"Test/Alt1", "Test/Alt2"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc
    assert registry._buffer_altn["Test/Alt1"] == dcc
    assert registry._buffer_altn["Test/Alt2"] == dcc


def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Test/DCC",
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


def test_register_duplicate_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"Test/Alt"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Test/Alt"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_conflict_with_existing_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Test/DCC1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_empty_altnames():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc
    assert len(registry._buffer_altn) == 0


def test_register_multiple_dcc():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"Test/Alt1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Test/Alt2"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    registry.register(dcc2)
    assert registry._buffer_main["Test/DCC1"] == dcc1
    assert registry._buffer_main["Test/DCC2"] == dcc2
    assert registry._buffer_altn["Test/Alt1"] == dcc1
    assert registry._buffer_altn["Test/Alt2"] == dcc2


# LLM-generated content at query #4
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #5
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


def test_last_payment_date_before_asof():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1)
    assert result == datetime.date(2014, 6, 1)


def test_last_payment_date_quarterly_frequency():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4)
    assert result == datetime.date(2015, 7, 7)


def test_last_payment_date_december_start():
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


def test_last_payment_date_with_eom_parameter():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 31), datetime.date(2015, 12, 31), 1, eom=31)
    assert result == datetime.date(2015, 1, 31)


def test_last_payment_date_eom_adjustment():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 31), datetime.date(2015, 2, 28), 1, eom=31)
    assert result == datetime.date(2015, 1, 31)


def test_last_payment_date_with_decimal_frequency():
    import datetime
    from decimal import Decimal
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), Decimal(1))
    assert result == datetime.date(2015, 1, 1)


def test_last_payment_date_quarterly_frequency_decimal():
    import datetime
    from decimal import Decimal
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), Decimal(4))
    assert result == datetime.date(2015, 7, 7)


# LLM-generated content at query #6
#--------------------------

```python
def test_register_raises_typeerror_when_dcc_name_already_registered():
    from decimal import Decimal
    from typing import Set
    
    # Create a mock DCFC function
    def mock_dcfc(start, asof, end, freq=None):
        return Decimal(0)
    
    # Create first DCC
    dcc1 = DCC(
        name="Act/Act",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    # Create second DCC with same name
    dcc2 = DCC(
        name="Act/Act",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    # Create registry and register first DCC
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    
    # Attempt to register second DCC with same name should raise TypeError
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError to be raised"
    except TypeError as e:
        assert "already registered" in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    dcc_registry = DCCRegistryMachinery()
    assert dcc_registry._buffer_main == {}
    assert dcc_registry._buffer_altn == {}


# LLM-generated content at query #8
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
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_act_act_one_day_non_leap():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 1, 1)
    asof = datetime.date(2007, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('365')


def test_dcfc_act_act_one_day_leap():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('366')


# LLM-generated content at query #9
#--------------------------

```python
def test_init_creates_empty_buffers():
    dcc_registry = DCCRegistryMachinery()
    assert dcc_registry._buffer_main == {}
    assert dcc_registry._buffer_altn == {}


# LLM-generated content at query #10
#--------------------------

```python
def test_init_creates_empty_buffers():
    machinery = DCCRegistryMachinery()
    assert machinery._buffer_main == {}
    assert machinery._buffer_altn == {}


# LLM-generated content at query #11
#--------------------------

```python
def test_last_payment_date_predicate_false():
    import datetime
    from decimal import Decimal
    
    # Test case where the predicate at line 1 evaluates to False
    # The predicate is: def _last_payment_date(start: Date, asof: Date, frequency: Union[int, Decimal], eom: Optional[int] = None) -> Date:
    # This tests that the function can be called with valid arguments
    
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = None
    
    # The function signature check - ensuring parameters are properly typed
    assert isinstance(start, datetime.date)
    assert isinstance(asof, datetime.date)
    assert isinstance(frequency, (int, Decimal))
    assert eom is None or isinstance(eom, int)


# LLM-generated content at query #12
#--------------------------

```python
def test_register_raises_error_when_altname_already_registered():
    from decimal import Decimal
    
    def mock_calculate_fraction(start, asof, end, freq=None):
        return Decimal(0)
    
    registry = DCCRegistryMachinery()
    
    dcc1 = DCC(
        name="Test1",
        altnames={"Alt1", "Alt2"},
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    dcc2 = DCC(
        name="Test2",
        altnames={"Alt1", "Alt3"},
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    registry.register(dcc1)
    
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError to be raised"
    except TypeError as e:
        assert "Day count convention 'Test2' is already registered" in str(e)


# LLM-generated content at query #13
#--------------------------

```python
def test_init_creates_empty_buffer_main():
    dcc_registry = DCCRegistryMachinery()
    assert dcc_registry._buffer_main == {}


# LLM-generated content at query #14
#--------------------------

```python
def test_dcc_registry_machinery_initialization():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #15
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


def test_is_last_day_of_month_april_last_day():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2024, 4, 30))
    assert result is True


def test_is_last_day_of_month_december_last_day():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2024, 12, 31))
    assert result is True


# LLM-generated content at query #16
#--------------------------

```python
def test_dcfc_act_act_predicate_false():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act
    
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 27)
    end = datetime.date(2008, 2, 27)
    
    result = dcfc_act_act(start, asof, end)
    
    assert result == Decimal('0')


# LLM-generated content at query #17
#--------------------------

```python
def test_find_with_exact_name():
    from decimal import Decimal
    import datetime
    
    machinery = DCCRegistryMachinery()
    dcc_mock = type('DCC', (), {'name': 'Act/Act', 'altnames': []})()
    machinery.register(dcc_mock)
    
    result = machinery.find('Act/Act')
    assert result == dcc_mock


def test_find_with_stripped_uppercase_name():
    from decimal import Decimal
    import datetime
    
    machinery = DCCRegistryMachinery()
    dcc_mock = type('DCC', (), {'name': 'ACT/ACT', 'altnames': []})()
    machinery.register(dcc_mock)
    
    result = machinery.find('  act/act  ')
    assert result == dcc_mock


def test_find_with_alternative_name():
    from decimal import Decimal
    import datetime
    
    machinery = DCCRegistryMachinery()
    dcc_mock = type('DCC', (), {'name': 'Actual/Actual', 'altnames': ['Act/Act']})()
    machinery.register(dcc_mock)
    
    result = machinery.find('Act/Act')
    assert result == dcc_mock


def test_find_with_alternative_name_stripped_uppercase():
    from decimal import Decimal
    import datetime
    
    machinery = DCCRegistryMachinery()
    dcc_mock = type('DCC', (), {'name': 'Actual/Actual', 'altnames': ['Act/Act']})()
    machinery.register(dcc_mock)
    
    result = machinery.find('  act/act  ')
    assert result == dcc_mock


def test_find_nonexistent_name():
    from decimal import Decimal
    import datetime
    
    machinery = DCCRegistryMachinery()
    dcc_mock = type('DCC', (), {'name': 'Act/Act', 'altnames': []})()
    machinery.register(dcc_mock)
    
    result = machinery.find('Nonexistent/Convention')
    assert result is None


def test_find_case_insensitive():
    from decimal import Decimal
    import datetime
    
    machinery = DCCRegistryMachinery()
    dcc_mock = type('DCC', (), {'name': 'ACT/ACT', 'altnames': []})()
    machinery.register(dcc_mock)
    
    result = machinery.find('act/act')
    assert result == dcc_mock


def test_find_with_whitespace():
    from decimal import Decimal
    import datetime
    
    machinery = DCCRegistryMachinery()
    dcc_mock = type('DCC', (), {'name': 'ACT/ACT', 'altnames': []})()
    machinery.register(dcc_mock)
    
    result = machinery.find('   ACT/ACT   ')
    assert result == dcc_mock


# LLM-generated content at query #18
#--------------------------

```python
def test_dcfc_30_e_360_example1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_e_360_example2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_e_360_example3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_e_360_example4():
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
    asof = datetime.date(2008, 2, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    expected = (15 - 30) + 30 * (2 - 1) + 360 * (2008 - 2008)
    assert result == Decimal(expected) / Decimal(360)


def test_dcfc_30_e_360_asof_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 2, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    expected = (30 - 15) + 30 * (2 - 1) + 360 * (2008 - 2008)
    assert result == Decimal(expected) / Decimal(360)


def test_dcfc_30_e_360_both_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 3, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    expected = (30 - 30) + 30 * (3 - 1) + 360 * (2008 - 2008)
    assert result == Decimal(expected) / Decimal(360)


def test_dcfc_30_e_360_same_date():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert result == Decimal(0)


def test_dcfc_30_e_360_year_difference():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2010, 1, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    expected = (15 - 15) + 30 * (1 - 1) + 360 * (2010 - 2008)
    assert result == Decimal(expected) / Decimal(360)


# LLM-generated content at query #19
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


def test_dcfc_act_365_a_over_year():
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


def test_dcfc_act_365_a_same_date():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    start = datetime.date(2020, 1, 1)
    result = dcfc_act_365_a(start=start, asof=start, end=start)
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
    asof = datetime.date(2020, 6, 30)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof, freq=Decimal('2'))
    assert result > Decimal('0')


# LLM-generated content at query #20
#--------------------------

```python
def test_dcfc_act_365_l_basic():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    
    result1 = round(dcfc_act_365_l(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14)
    result2 = round(dcfc_act_365_l(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14)
    result3 = round(dcfc_act_365_l(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14)
    result4 = round(dcfc_act_365_l(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14)
    
    assert result1 == Decimal('0.16939890710383')
    assert result2 == Decimal('0.17213114754098')
    assert result3 == Decimal('1.08196721311475')
    assert result4 == Decimal('1.32876712328767')


def test_dcfc_act_365_l_same_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2007, 12, 28)
    end = datetime.date(2007, 12, 28)
    
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    assert result == Decimal('0')


def test_dcfc_act_365_l_one_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    start = datetime.date(2007, 1, 1)
    asof = datetime.date(2007, 1, 2)
    end = datetime.date(2007, 1, 2)
    
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    assert result == Decimal('1') / Decimal('365')


def test_dcfc_act_365_l_leap_year():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    assert result == Decimal('28') / Decimal('366')


def test_dcfc_act_365_l_with_freq_parameter():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    freq = Decimal('2')
    
    result = round(dcfc_act_365_l(start=start, asof=asof, end=end, freq=freq), 14)
    assert result == Decimal('0.16939890710383')


# LLM-generated content at query #21
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


def test_last_payment_date_annual_frequency_june_start():
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


# LLM-generated content at query #22
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


def test_is_last_day_of_month_april_30_days():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2024, 4, 30))
    assert result is True


def test_is_last_day_of_month_december_last():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2024, 12, 31))
    assert result is True


# LLM-generated content at query #23
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


def test_is_last_day_of_month_april_30():
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


def test_is_last_day_of_month_first_day():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2023, 6, 1))
    assert result is False


# LLM-generated content at query #24
#--------------------------

```python
def test_find_with_stripped_uppercase_fallback():
    from unittest.mock import Mock
    
    # Create a mock DCC object
    mock_dcc = Mock()
    mock_dcc.name = "ACT/ACT"
    mock_dcc.altnames = []
    
    # Create registry and register the mock DCC
    registry = DCCRegistryMachinery()
    registry._buffer_main["ACT/ACT"] = mock_dcc
    
    # Test that find with lowercase and spaces falls back to stripped uppercase
    result = registry.find("  act/act  ")
    
    assert result is mock_dcc
    assert result.name == "ACT/ACT"


# LLM-generated content at query #25
#--------------------------

```python
def test_has_leap_day_with_leap_day_in_range():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 2, 28)
    end = datetime.date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_without_leap_day_in_range():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2019, 1, 1)
    end = datetime.date(2019, 12, 31)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_with_leap_day_at_start():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_with_leap_day_at_end():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 2, 1)
    end = datetime.date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_multiple_leap_years_in_range():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2019, 1, 1)
    end = datetime.date(2021, 12, 31)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_leap_day_before_range():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 12, 31)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_leap_day_after_range():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 2, 28)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_single_day_leap_day():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_single_day_non_leap_day():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2019, 2, 28)
    end = datetime.date(2019, 2, 28)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_century_leap_year():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2000, 2, 28)
    end = datetime.date(2000, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


# LLM-generated content at query #26
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


def test_dcfc_30_360_us_same_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_30_360_us_one_day_difference():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 16)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('360')


def test_dcfc_30_360_us_month_end_handling():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert result == Decimal('29') / Decimal('360')


def test_dcfc_30_360_us_with_freq_parameter():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 6, 30)
    freq = Decimal('2')
    result = dcfc_30_360_us(start=start, asof=asof, end=asof, freq=freq)
    assert result == Decimal('180') / Decimal('360')


# LLM-generated content at query #27
#--------------------------

```python
def test_dcc_registry_machinery_init():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #28
#--------------------------

```python
def test_dcfc_30_e_360_asof_day_not_31():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    
    assert asof.day != 31
    assert result == Decimal('60') / Decimal('360')


# LLM-generated content at query #29
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
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_altname_conflicts_with_existing_main_name():
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


def test_register_altname_conflicts_with_existing_altname():
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
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_multiple_dcc_sequential():
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


# LLM-generated content at query #30
#--------------------------

```python
def test_init_creates_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_nl_365_basic():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')


def test_dcfc_nl_365_leap_day():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16986301369863')


def test_dcfc_nl_365_longer_period():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08219178082192')


def test_dcfc_nl_365_another_period():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32602739726027')


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


def test_dcfc_nl_365_with_freq_parameter():
    import datetime
    from decimal import Decimal
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 15)
    freq = Decimal('4')
    result = dcfc_nl_365(start=start, asof=asof, end=asof, freq=freq)
    assert result == Decimal('14') / Decimal('365')


# LLM-generated content at query #2
#--------------------------

```python
def test_find_with_exact_main_name():
    from itertools import chain
    
    class DCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["Actual/Actual"])
    registry.register(dcc)
    
    result = registry.find("Act/Act")
    assert result is dcc


def test_find_with_exact_alternative_name():
    from itertools import chain
    
    class DCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["Actual/Actual"])
    registry.register(dcc)
    
    result = registry.find("Actual/Actual")
    assert result is dcc


def test_find_with_stripped_uppercase_name():
    from itertools import chain
    
    class DCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = DCC("ACT/ACT", [])
    registry.register(dcc)
    
    result = registry.find("  act/act  ")
    assert result is dcc


def test_find_with_nonexistent_name():
    from itertools import chain
    
    class DCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", [])
    registry.register(dcc)
    
    result = registry.find("NonExistent")
    assert result is None


def test_find_with_lowercase_variant():
    from itertools import chain
    
    class DCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = DCC("30/360 US", [])
    registry.register(dcc)
    
    result = registry.find("30/360 us")
    assert result is dcc


def test_find_with_whitespace_and_case_variation():
    from itertools import chain
    
    class DCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = DCC("ACT/365F", ["Act/365 Fixed"])
    registry.register(dcc)
    
    result = registry.find("  act/365f  ")
    assert result is dcc


def test_find_alternative_name_with_case_and_whitespace():
    from itertools import chain
    
    class DCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = DCC("ACT/365F", ["Act/365 Fixed"])
    registry.register(dcc)
    
    result = registry.find("  act/365 fixed  ")
    assert result is dcc


# LLM-generated content at query #3
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_main) == 0
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #4
#--------------------------

```python
def test_has_leap_day_with_leap_day_in_range():
    import datetime
    import calendar
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 2, 1)
    end = datetime.date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_without_leap_day_in_range():
    import datetime
    import calendar
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 12, 31)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_multiple_leap_years_with_leap_day():
    import datetime
    import calendar
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2024, 2, 29)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_before_leap_day():
    import datetime
    import calendar
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 2, 28)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_after_leap_day():
    import datetime
    import calendar
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 12, 31)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_exact_leap_day_range():
    import datetime
    import calendar
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_spanning_multiple_leap_years():
    import datetime
    import calendar
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2019, 1, 1)
    end = datetime.date(2024, 12, 31)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_no_leap_years_in_range():
    import datetime
    import calendar
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2022, 12, 31)
    result = _has_leap_day(start, end)
    assert result is False


# LLM-generated content at query #5
#--------------------------

```python
def test_register_new_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc


def test_register_dcc_with_altnames():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames={"ALT1", "ALT2"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc
    assert registry._buffer_altn["ALT1"] == dcc
    assert registry._buffer_altn["ALT2"] == dcc


def test_register_duplicate_main_name_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Test/DCC",
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


def test_register_duplicate_altname_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"SHARED"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"SHARED"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_altname_conflicts_with_main_name_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Test/DCC1"},
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
        name="Test/DCC1",
        altnames={"ALT1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"ALT2"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    registry.register(dcc2)
    assert registry._buffer_main["Test/DCC1"] == dcc1
    assert registry._buffer_main["Test/DCC2"] == dcc2
    assert registry._buffer_altn["ALT1"] == dcc1
    assert registry._buffer_altn["ALT2"] == dcc2


# LLM-generated content at query #6
#--------------------------

```python
def test_register_duplicate_main_name_raises_typeerror():
    from decimal import Decimal
    
    def dummy_calculate_fraction_method(start, asof, end, freq=None):
        return Decimal(0)
    
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Act/Act",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction_method
    )
    dcc2 = DCC(
        name="Act/Act",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction_method
    )
    
    registry.register(dcc1)
    
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError to be raised"
    except TypeError as e:
        assert "Day count convention 'Act/Act' is already registered" in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_dcfc_act_365_l():
    import datetime
    from decimal import Decimal
    
    # Test case 1: Same day should return 0
    result1 = dcfc_act_365_l(datetime.date(2017, 1, 1), datetime.date(2017, 1, 1), datetime.date(2017, 1, 1))
    assert result1 == Decimal('0')
    
    # Test case 2: One day difference in non-leap year
    result2 = dcfc_act_365_l(datetime.date(2017, 1, 1), datetime.date(2017, 1, 2), datetime.date(2017, 1, 2))
    assert result2 == Decimal('1') / Decimal('365')
    
    # Test case 3: Example from docstring - ex1
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result3 = dcfc_act_365_l(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result3, 14) == Decimal('0.16939890710383')
    
    # Test case 4: Example from docstring - ex2
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result4 = dcfc_act_365_l(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result4, 14) == Decimal('0.17213114754098')
    
    # Test case 5: Example from docstring - ex3
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result5 = dcfc_act_365_l(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result5, 14) == Decimal('1.08196721311475')
    
    # Test case 6: Example from docstring - ex4
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result6 = dcfc_act_365_l(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result6, 14) == Decimal('1.32876712328767')
    
    # Test case 7: Leap year divisor (366 days)
    leap_start = datetime.date(2020, 1, 1)
    leap_asof = datetime.date(2020, 1, 2)
    result7 = dcfc_act_365_l(start=leap_start, asof=leap_asof, end=leap_asof)
    assert result7 == Decimal('1') / Decimal('366')
    
    # Test case 8: Non-leap year divisor (365 days)
    non_leap_start = datetime.date(2019, 1, 1)
    non_leap_asof = datetime.date(2019, 1, 2)
    result8 = dcfc_act_365_l(start=non_leap_start, asof=non_leap_asof, end=non_leap_asof)
    assert result8 == Decimal('1') / Decimal('365')


# LLM-generated content at query #8
#--------------------------

```python
def test_init_creates_empty_buffers():
    dcc_registry = DCCRegistryMachinery()
    assert dcc_registry._buffer_main == {}
    assert dcc_registry._buffer_altn == {}


# LLM-generated content at query #9
#--------------------------

```python
def test_init_creates_empty_buffers():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert len(registry._buffer_main) == 0
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #10
#--------------------------

```python
def test_interest_calculates_accrued_interest_correctly():
    from decimal import Decimal
    from datetime import date
    
    def mock_calculate_fraction(start, asof, end, freq=None):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    principal = Decimal("1000")
    rate = Decimal("0.05")
    start = date(2024, 1, 1)
    asof = date(2024, 6, 30)
    end = date(2024, 12, 31)
    
    result = dcc.interest(principal, rate, start, asof, end)
    
    expected = principal * rate * Decimal("0.5")
    assert result == expected


def test_interest_uses_asof_as_end_when_end_is_none():
    from decimal import Decimal
    from datetime import date
    
    def mock_calculate_fraction(start, asof, end, freq=None):
        return Decimal("0.25")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    principal = Decimal("2000")
    rate = Decimal("0.1")
    start = date(2024, 1, 1)
    asof = date(2024, 3, 31)
    
    result = dcc.interest(principal, rate, start, asof, end=None)
    
    expected = principal * rate * Decimal("0.25")
    assert result == expected


def test_interest_with_frequency_parameter():
    from decimal import Decimal
    from datetime import date
    
    call_args = []
    
    def mock_calculate_fraction(start, asof, end, freq=None):
        call_args.append((start, asof, end, freq))
        return Decimal("0.75")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    principal = Decimal("5000")
    rate = Decimal("0.02")
    start = date(2024, 1, 1)
    asof = date(2024, 9, 30)
    end = date(2024, 12, 31)
    freq = Decimal("4")
    
    result = dcc.interest(principal, rate, start, asof, end, freq)
    
    expected = principal * rate * Decimal("0.75")
    assert result == expected
    assert call_args[0][3] == freq


def test_interest_with_zero_rate():
    from decimal import Decimal
    from datetime import date
    
    def mock_calculate_fraction(start, asof, end, freq=None):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    principal = Decimal("1000")
    rate = Decimal("0")
    start = date(2024, 1, 1)
    asof = date(2024, 6, 30)
    end = date(2024, 12, 31)
    
    result = dcc.interest(principal, rate, start, asof, end)
    
    assert result == Decimal("0")


# LLM-generated content at query #11
#--------------------------

```python
def test_coupon_basic():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal('0.5')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    principal = Decimal('1000')
    rate = Decimal('0.05')
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 6, 15)
    end = datetime.date(2015, 1, 1)
    freq = 2
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = principal * rate * Decimal('0.5')
    assert result == expected


def test_coupon_with_eom():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal('0.25')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    principal = Decimal('5000')
    rate = Decimal('0.10')
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2015, 12, 31)
    end = datetime.date(2016, 1, 6)
    freq = 2
    eom = 15
    
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal('0.25')
    assert result == expected


def test_coupon_annual_frequency():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal('0.75')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    principal = Decimal('10000')
    rate = Decimal('0.02')
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 12, 9)
    end = datetime.date(2015, 12, 4)
    freq = 1
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = principal * rate * Decimal('0.75')
    assert result == expected


def test_coupon_quarterly_frequency():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal('0.33')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    principal = Decimal('2000')
    rate = Decimal('0.08')
    start = datetime.date(2008, 7, 7)
    asof = datetime.date(2015, 10, 6)
    end = datetime.date(2016, 1, 1)
    freq = 4
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = principal * rate * Decimal('0.33')
    assert result == expected


def test_coupon_with_decimal_frequency():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal('0.5')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    principal = Decimal('7500')
    rate = Decimal('0.06')
    start = datetime.date(2014, 6, 1)
    asof = datetime.date(2015, 4, 30)
    end = datetime.date(2016, 1, 1)
    freq = Decimal('2')
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = principal * rate * Decimal('0.5')
    assert result == expected


# LLM-generated content at query #12
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    machinery = DCCRegistryMachinery()
    assert machinery._buffer_main == {}
    assert machinery._buffer_altn == {}


# LLM-generated content at query #13
#--------------------------

```python
def test_init_creates_empty_main_buffer():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}


# LLM-generated content at query #14
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
    
    assert result == [date(2023, 1, 1)]


def test_get_date_range_different_months():
    from datetime import date
    from pypara.dcc import _get_date_range
    
    start = date(2023, 1, 30)
    end = date(2023, 2, 2)
    result = list(_get_date_range(start, end))
    
    expected = [date(2023, 1, 30), date(2023, 1, 31), date(2023, 2, 1)]
    assert result == expected


def test_get_date_range_different_years():
    from datetime import date
    from pypara.dcc import _get_date_range
    
    start = date(2022, 12, 31)
    end = date(2023, 1, 2)
    result = list(_get_date_range(start, end))
    
    expected = [date(2022, 12, 31), date(2023, 1, 1)]
    assert result == expected


# LLM-generated content at query #15
#--------------------------

```python
def test_register_valid_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/360",
        altnames={"Test360", "TestDCC"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal('0.5')
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/360"] == dcc
    assert registry._buffer_altn["Test360"] == dcc
    assert registry._buffer_altn["TestDCC"] == dcc


def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Act/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal('0.5')
    )
    dcc2 = DCC(
        name="Act/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal('0.6')
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_duplicate_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Act/360",
        altnames={"Actual/360"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal('0.5')
    )
    dcc2 = DCC(
        name="Actual/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal('0.6')
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_altname_conflicts_with_existing_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Act/360",
        altnames={"Actual/360"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal('0.5')
    )
    dcc2 = DCC(
        name="Actual/365",
        altnames={"Actual/360"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal('0.6')
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_multiple_valid_dcc():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Act/360",
        altnames={"Actual/360"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal('0.5')
    )
    dcc2 = DCC(
        name="Act/365",
        altnames={"Actual/365"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal('0.6')
    )
    registry.register(dcc1)
    registry.register(dcc2)
    assert len(registry._buffer_main) == 2
    assert len(registry._buffer_altn) == 2
    assert registry._buffer_main["Act/360"] == dcc1
    assert registry._buffer_main["Act/365"] == dcc2


# LLM-generated content at query #16
#--------------------------

```python
def test_register_raises_error_when_altname_already_registered():
    from decimal import Decimal
    
    # Create a mock DCFC function
    def mock_dcfc(start, asof, end, freq=None):
        return Decimal(0)
    
    # Create first DCC with altname "ALT1"
    dcc1 = DCC(
        name="DCC1",
        altnames={"ALT1"},
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    # Create second DCC with altname "ALT1" (same as dcc1's altname)
    dcc2 = DCC(
        name="DCC2",
        altnames={"ALT1"},
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    registry = DCCRegistryMachinery()
    
    # Register first DCC successfully
    registry.register(dcc1)
    
    # Attempting to register second DCC with conflicting altname should raise TypeError
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError to be raised"
    except TypeError as e:
        assert "already registered" in str(e)


# LLM-generated content at query #17
#--------------------------

```python
def test_register_valid_dcc():
    registry = DCCRegistryMachinery()
    
    def dummy_calculate_fraction(start, asof, end, freq=None):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test/DCC",
        altnames={"Test/Alternative", "T/D"},
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    registry.register(dcc)
    
    assert registry._buffer_main["Test/DCC"] == dcc
    assert registry._buffer_altn["Test/Alternative"] == dcc
    assert registry._buffer_altn["T/D"] == dcc


def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    
    def dummy_calculate_fraction(start, asof, end, freq=None):
        return Decimal("0.5")
    
    dcc1 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    dcc2 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    registry.register(dcc1)
    
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError to be raised"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_duplicate_altname():
    registry = DCCRegistryMachinery()
    
    def dummy_calculate_fraction(start, asof, end, freq=None):
        return Decimal("0.5")
    
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"Test/Alternative"},
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Test/Alternative"},
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    registry.register(dcc1)
    
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError to be raised"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_altname_conflicts_with_main_name():
    registry = DCCRegistryMachinery()
    
    def dummy_calculate_fraction(start, asof, end, freq=None):
        return Decimal("0.5")
    
    dcc1 = DCC(
        name="Test/DCC1",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Test/DCC1"},
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    registry.register(dcc1)
    
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError to be raised"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_multiple_altnames():
    registry = DCCRegistryMachinery()
    
    def dummy_calculate_fraction(start, asof, end, freq=None):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test/DCC",
        altnames={"Alt1", "Alt2", "Alt3"},
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    registry.register(dcc)
    
    assert registry._buffer_main["Test/DCC"] == dcc
    assert registry._buffer_altn["Alt1"] == dcc
    assert registry._buffer_altn["Alt2"] == dcc
    assert registry._buffer_altn["Alt3"] == dcc
    assert len(registry._buffer_main) == 1
    assert len(registry._buffer_altn) == 3


# LLM-generated content at query #18
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


def test_last_payment_date_annual_frequency_june_start():
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


def test_last_payment_date_semi_annual_frequency_december_31():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 12, 15)


def test_last_payment_date_with_eom_parameter():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 15), datetime.date(2015, 12, 31), 1, eom=15)
    assert result == datetime.date(2015, 1, 15)


def test_last_payment_date_before_first_payment():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2014, 3, 31), 1)
    assert result == datetime.date(2014, 6, 1)


def test_last_payment_date_eom_adjustment():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 31), datetime.date(2014, 2, 28), 1)
    assert result == datetime.date(2014, 1, 31)


# LLM-generated content at query #19
#--------------------------

```python
def test_coupon_basic():
    import datetime
    from decimal import Decimal
    
    dcc = DCC(
        name="Actual/Actual",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    
    principal = Decimal("1000")
    rate = Decimal("0.05")
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 6, 15)
    end = datetime.date(2015, 1, 1)
    freq = 2
    eom = None
    
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Decimal("25")


def test_coupon_with_eom():
    import datetime
    from decimal import Decimal
    
    dcc = DCC(
        name="30/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.25")
    )
    
    principal = Decimal("10000")
    rate = Decimal("0.04")
    start = datetime.date(2014, 1, 31)
    asof = datetime.date(2014, 3, 15)
    end = datetime.date(2014, 7, 31)
    freq = 4
    eom = 31
    
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Decimal("100")


def test_coupon_annual_frequency():
    import datetime
    from decimal import Decimal
    
    dcc = DCC(
        name="Actual/365",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.75")
    )
    
    principal = Decimal("5000")
    rate = Decimal("0.06")
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 10, 1)
    end = datetime.date(2015, 1, 1)
    freq = 1
    eom = None
    
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Decimal("225")


def test_coupon_semi_annual():
    import datetime
    from decimal import Decimal
    
    dcc = DCC(
        name="Actual/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.3")
    )
    
    principal = Decimal("2000")
    rate = Decimal("0.08")
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2015, 12, 31)
    end = datetime.date(2016, 6, 15)
    freq = 2
    eom = 15
    
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Decimal("48")


# LLM-generated content at query #20
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #21
#--------------------------

```python
def test_has_leap_day_no_leap_day_in_range():
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
    
    start = date(2021, 3, 1)
    end = date(2021, 12, 31)
    result = _has_leap_day(start, end)
    assert result == False


# LLM-generated content at query #22
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


def test_dcfc_act_365_a_long_period_1():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08196721311475')


def test_dcfc_act_365_a_long_period_2():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32513661202186')


def test_dcfc_act_365_a_same_date():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    start = datetime.date(2020, 1, 1)
    result = dcfc_act_365_a(start=start, asof=start, end=start)
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
    
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    freq = Decimal('2')
    result = dcfc_act_365_a(start=start, asof=asof, end=asof, freq=freq)
    assert round(result, 14) == Decimal('0.16986301369863')


# LLM-generated content at query #23
#--------------------------

```python
def test_register_raises_typeerror_when_altname_already_registered():
    from decimal import Decimal
    
    def dummy_calculate_fraction(start, asof, end, freq=None):
        return Decimal(0)
    
    registry = DCCRegistryMachinery()
    
    dcc1 = DCC(
        name="Test1",
        altnames={"Alt1", "Alt2"},
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    dcc2 = DCC(
        name="Test2",
        altnames={"Alt1", "Alt3"},
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    registry.register(dcc1)
    
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError to be raised"
    except TypeError as e:
        assert "already registered" in str(e)


# LLM-generated content at query #24
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
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_act_act_one_day_non_leap():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 1, 1)
    asof = datetime.date(2007, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('365')


def test_dcfc_act_act_one_day_leap():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('366')


# LLM-generated content at query #25
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


def test_dcfc_act_act_icma_with_freq():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal('2')
    
    result = dcfc_act_act_icma(start, asof, end, freq)
    expected = Decimal(_get_actual_day_count(start, asof)) / Decimal(_get_actual_day_count(start, end)) / freq
    assert result == expected


def test_dcfc_act_act_icma_one_day_period():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 3, 3)
    end = datetime.date(2019, 3, 3)
    
    result = dcfc_act_act_icma(start, asof, end)
    assert result == Decimal('1')


def test_dcfc_act_act_icma_half_period():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 7, 1)
    end = datetime.date(2020, 1, 1)
    
    result = dcfc_act_act_icma(start, asof, end)
    assert round(result, 2) == Decimal('0.50')


def test_dcfc_act_act_icma_with_freq_4():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal('4')
    
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert result == round(result, 10)


# LLM-generated content at query #26
#--------------------------

```python
def test_next_payment_date_annual_frequency():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, None)
    assert result == datetime.date(2015, 1, 1)


def test_next_payment_date_annual_frequency_with_eom():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, 15)
    assert result == datetime.date(2015, 1, 15)


def test_next_payment_date_semi_annual_frequency():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 2, None)
    assert result == datetime.date(2014, 7, 1)


def test_next_payment_date_quarterly_frequency():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 4, None)
    assert result == datetime.date(2014, 4, 1)


def test_next_payment_date_monthly_frequency():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 12, None)
    assert result == datetime.date(2014, 2, 1)


def test_next_payment_date_with_eom_february():
    import datetime
    from dateutil.relativedelta import relativedelta
    from decimal import Decimal
    
    result = _next_payment_date(datetime.date(2014, 1, 31), Decimal(1), 31)
    assert result == datetime.date(2015, 1, 31)


def test_next_payment_date_eom_invalid_day():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 31), 1, 31)
    assert result == datetime.date(2015, 1, 31)


def test_next_payment_date_with_decimal_frequency():
    import datetime
    from dateutil.relativedelta import relativedelta
    from decimal import Decimal
    
    result = _next_payment_date(datetime.date(2014, 1, 1), Decimal(2), None)
    assert result == datetime.date(2014, 7, 1)


# LLM-generated content at query #27
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


def test_dcfc_act_365_a_year_span():
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


def test_dcfc_act_365_a_same_date():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    start = datetime.date(2020, 1, 1)
    result = dcfc_act_365_a(start=start, asof=start, end=start)
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
    asof = datetime.date(2020, 2, 1)
    freq = Decimal('4')
    result = dcfc_act_365_a(start=start, asof=asof, end=asof, freq=freq)
    assert result == Decimal('31') / Decimal('365')


# LLM-generated content at query #28
#--------------------------

```python
def test_coupon_basic_annual_frequency():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    principal = Decimal("1000")
    rate = Decimal("0.05")
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 6, 1)
    end = datetime.date(2015, 1, 1)
    freq = 1
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal("25")


def test_coupon_with_eom():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.25")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    principal = Decimal("2000")
    rate = Decimal("0.04")
    start = datetime.date(2014, 1, 31)
    asof = datetime.date(2014, 3, 15)
    end = datetime.date(2014, 7, 31)
    freq = 2
    eom = 31
    
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Decimal("20")


def test_coupon_semi_annual_frequency():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.75")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    principal = Decimal("5000")
    rate = Decimal("0.06")
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2016, 1, 6)
    end = datetime.date(2016, 6, 15)
    freq = 2
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal("1125")


def test_coupon_quarterly_frequency():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.1")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    principal = Decimal("10000")
    rate = Decimal("0.08")
    start = datetime.date(2008, 7, 7)
    asof = datetime.date(2015, 10, 6)
    end = datetime.date(2016, 1, 7)
    freq = 4
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal("80")


def test_coupon_with_decimal_frequency():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    principal = Decimal("3000")
    rate = Decimal("0.03")
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 9, 1)
    end = datetime.date(2015, 1, 1)
    freq = Decimal("1")
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal("45")


def test_coupon_with_none_eom():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.33")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    principal = Decimal("1500")
    rate = Decimal("0.07")
    start = datetime.date(2014, 6, 1)
    asof = datetime.date(2015, 4, 30)
    end = datetime.date(2015, 6, 1)
    freq = 1
    eom = None
    
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Decimal("346.5")


# LLM-generated content at query #29
#--------------------------

```python
def test_calculate_daily_fraction_basic():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal('0.1')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start_date = date(2023, 1, 1)
    asof_date = date(2023, 1, 3)
    end_date = date(2023, 12, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    expected = Decimal('0')
    assert result == expected


def test_calculate_daily_fraction_with_different_fractions():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    fractions = {
        (date(2023, 1, 1), date(2023, 1, 2), date(2023, 12, 31)): Decimal('0.05'),
        (date(2023, 1, 1), date(2023, 1, 3), date(2023, 12, 31)): Decimal('0.15')
    }
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return fractions.get((start, asof, end), Decimal('0'))
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start_date = date(2023, 1, 1)
    asof_date = date(2023, 1, 3)
    end_date = date(2023, 12, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    expected = Decimal('0.1')
    assert result == expected


def test_calculate_daily_fraction_asof_equals_start():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        if asof == start:
            return Decimal('0.05')
        return Decimal('0')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start_date = date(2023, 1, 1)
    asof_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    expected = Decimal('0.05')
    assert result == expected


def test_calculate_daily_fraction_with_freq():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        if freq is not None:
            return Decimal('0.25')
        return Decimal('0.1')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start_date = date(2023, 1, 1)
    asof_date = date(2023, 1, 3)
    end_date = date(2023, 12, 31)
    freq = Decimal('4')
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date, freq)
    expected = Decimal('0')
    assert result == expected


def test_calculate_daily_fraction_returns_decimal():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal('0.123456')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start_date = date(2023, 1, 1)
    asof_date = date(2023, 1, 3)
    end_date = date(2023, 12, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    assert isinstance(result, Decimal)


# LLM-generated content at query #30
#--------------------------

```python
def test_dcfc_30_360_us_example_1():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    rounded_result = round(result, 14)
    expected = Decimal('0.16666666666667')
    assert rounded_result == expected


def test_dcfc_30_360_us_example_2():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    rounded_result = round(result, 14)
    expected = Decimal('0.16944444444444')
    assert rounded_result == expected


def test_dcfc_30_360_us_example_3():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    rounded_result = round(result, 14)
    expected = Decimal('1.08333333333333')
    assert rounded_result == expected


def test_dcfc_30_360_us_example_4():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    rounded_result = round(result, 14)
    expected = Decimal('1.33333333333333')
    assert rounded_result == expected


def test_dcfc_30_360_us_same_dates():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    end = datetime.date(2008, 1, 15)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal('0')
    assert result == expected


def test_dcfc_30_360_us_one_day_difference():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 16)
    end = datetime.date(2008, 1, 16)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal('1') / Decimal('360')
    assert result == expected


def test_dcfc_30_360_us_month_end_adjustment():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal('29') / Decimal('360')
    assert result == expected


def test_dcfc_30_360_us_year_change():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2007, 12, 31)
    asof = datetime.date(2008, 1, 1)
    end = datetime.date(2008, 1, 1)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal('1') / Decimal('360')
    assert result == expected


def test_dcfc_30_360_us_d2_31_with_d1_30():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2008, 1, 30)
    asof = datetime.date(2008, 1, 31)
    end = datetime.date(2008, 1, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal('0') / Decimal('360')
    assert result == expected


def test_dcfc_30_360_us_multiple_months():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 3, 1)
    end = datetime.date(2008, 3, 1)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal('60') / Decimal('360')
    assert result == expected


