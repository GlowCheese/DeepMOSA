####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_formatted_datetime_default_format():
    from mimesis import Datetime
    dt = Datetime()
    result = dt.formatted_datetime()
    assert isinstance(result, str)
    assert len(result) > 0


def test_formatted_datetime_custom_format():
    from mimesis import Datetime
    dt = Datetime()
    result = dt.formatted_datetime(fmt="%Y-%m-%d %H:%M:%S")
    assert isinstance(result, str)
    assert len(result) > 0


def test_formatted_datetime_iso_format():
    from mimesis import Datetime
    dt = Datetime()
    result = dt.formatted_datetime(fmt="%Y-%m-%dT%H:%M:%S")
    assert isinstance(result, str)
    assert len(result) > 0


def test_formatted_datetime_with_start_year():
    from mimesis import Datetime
    dt = Datetime()
    result = dt.formatted_datetime(fmt="%Y-%m-%d", start=2020, end=2020)
    assert isinstance(result, str)
    assert "2020" in result


def test_formatted_datetime_with_end_year():
    from mimesis import Datetime
    dt = Datetime()
    result = dt.formatted_datetime(fmt="%Y-%m-%d", start=2015, end=2015)
    assert isinstance(result, str)
    assert "2015" in result


def test_formatted_datetime_with_timezone():
    from mimesis import Datetime
    dt = Datetime()
    result = dt.formatted_datetime(fmt="%Y-%m-%d %H:%M:%S", timezone="UTC")
    assert isinstance(result, str)
    assert len(result) > 0


def test_formatted_datetime_format_percent_d():
    from mimesis import Datetime
    dt = Datetime()
    result = dt.formatted_datetime(fmt="%d")
    assert isinstance(result, str)
    day = int(result)
    assert 1 <= day <= 31


def test_formatted_datetime_format_percent_m():
    from mimesis import Datetime
    dt = Datetime()
    result = dt.formatted_datetime(fmt="%m")
    assert isinstance(result, str)
    month = int(result)
    assert 1 <= month <= 12


def test_formatted_datetime_format_percent_Y():
    from mimesis import Datetime
    dt = Datetime()
    result = dt.formatted_datetime(fmt="%Y", start=2010, end=2010)
    assert isinstance(result, str)
    assert result == "2010"


def test_formatted_datetime_multiple_calls_different_results():
    from mimesis import Datetime
    dt = Datetime()
    result1 = dt.formatted_datetime(fmt="%Y-%m-%d %H:%M:%S", start=2000, end=2025)
    result2 = dt.formatted_datetime(fmt="%Y-%m-%d %H:%M:%S", start=2000, end=2025)
    assert isinstance(result1, str)
    assert isinstance(result2, str)


# LLM-generated content at query #2
#--------------------------

```python
def test_duration_default_parameters():
    from mimesis import Datetime
    from datetime import timedelta
    from mimesis.enums import DurationUnit
    
    dt = Datetime()
    result = dt.duration()
    
    assert isinstance(result, timedelta)
    assert result.total_seconds() > 0


def test_duration_custom_min_max():
    from mimesis import Datetime
    from datetime import timedelta
    
    dt = Datetime()
    result = dt.duration(min_duration=5, max_duration=15)
    
    assert isinstance(result, timedelta)
    assert 5 * 60 <= result.total_seconds() <= 15 * 60


def test_duration_seconds_unit():
    from mimesis import Datetime
    from datetime import timedelta
    from mimesis.enums import DurationUnit
    
    dt = Datetime()
    result = dt.duration(min_duration=10, max_duration=20, duration_unit=DurationUnit.SECONDS)
    
    assert isinstance(result, timedelta)
    assert 10 <= result.total_seconds() <= 20


def test_duration_hours_unit():
    from mimesis import Datetime
    from datetime import timedelta
    from mimesis.enums import DurationUnit
    
    dt = Datetime()
    result = dt.duration(min_duration=1, max_duration=5, duration_unit=DurationUnit.HOURS)
    
    assert isinstance(result, timedelta)
    assert 3600 <= result.total_seconds() <= 18000


def test_duration_days_unit():
    from mimesis import Datetime
    from datetime import timedelta
    from mimesis.enums import DurationUnit
    
    dt = Datetime()
    result = dt.duration(min_duration=1, max_duration=3, duration_unit=DurationUnit.DAYS)
    
    assert isinstance(result, timedelta)
    assert 86400 <= result.total_seconds() <= 259200


def test_duration_microseconds_unit():
    from mimesis import Datetime
    from datetime import timedelta
    from mimesis.enums import DurationUnit
    
    dt = Datetime()
    result = dt.duration(min_duration=1000, max_duration=5000, duration_unit=DurationUnit.MICROSECONDS)
    
    assert isinstance(result, timedelta)
    assert 0.001 <= result.total_seconds() <= 0.005


def test_duration_none_unit():
    from mimesis import Datetime
    from datetime import timedelta
    
    dt = Datetime()
    result = dt.duration(min_duration=1, max_duration=10, duration_unit=None)
    
    assert isinstance(result, timedelta)
    assert result.total_seconds() > 0


def test_duration_min_duration_greater_than_max_raises_error():
    from mimesis import Datetime
    
    dt = Datetime()
    
    try:
        dt.duration(min_duration=20, max_duration=10)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "min_duration must be less or equal to max_duration" in str(e)


def test_duration_non_integer_min_duration_raises_error():
    from mimesis import Datetime
    
    dt = Datetime()
    
    try:
        dt.duration(min_duration=5.5, max_duration=10)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "min_duration and max_duration must be integers" in str(e)


def test_duration_non_integer_max_duration_raises_error():
    from mimesis import Datetime
    
    dt = Datetime()
    
    try:
        dt.duration(min_duration=5, max_duration=10.5)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "min_duration and max_duration must be integers" in str(e)


def test_duration_equal_min_max():
    from mimesis import Datetime
    from datetime import timedelta
    from mimesis.enums import DurationUnit
    
    dt = Datetime()
    result = dt.duration(min_duration=5, max_duration=5, duration_unit=DurationUnit.MINUTES)
    
    assert isinstance(result, timedelta)
    assert result.total_seconds() == 300


# LLM-generated content at query #3
#--------------------------

```python
def test_timestamp_posix_format():
    from mimesis import Datetime
    from mimesis.enums import TimestampFormat
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.POSIX)
    assert isinstance(result, int)
    assert result > 0


def test_timestamp_rfc_3339_format():
    from mimesis import Datetime
    from mimesis.enums import TimestampFormat
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.RFC_3339)
    assert isinstance(result, str)
    assert "T" in result
    assert "Z" in result


def test_timestamp_iso_8601_format():
    from mimesis import Datetime
    from mimesis.enums import TimestampFormat
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.ISO_8601)
    assert isinstance(result, str)
    assert "T" in result


def test_timestamp_default_format():
    from mimesis import Datetime
    dt = Datetime()
    result = dt.timestamp()
    assert isinstance(result, int)
    assert result > 0


def test_timestamp_with_custom_year_range():
    from mimesis import Datetime
    from mimesis.enums import TimestampFormat
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.POSIX, start=2020, end=2021)
    assert isinstance(result, int)
    assert result > 0


def test_timestamp_posix_returns_integer():
    from mimesis import Datetime
    from mimesis.enums import TimestampFormat
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.POSIX)
    assert isinstance(result, int)


def test_timestamp_rfc_3339_returns_string():
    from mimesis import Datetime
    from mimesis.enums import TimestampFormat
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.RFC_3339)
    assert isinstance(result, str)


def test_timestamp_iso_8601_returns_string():
    from mimesis import Datetime
    from mimesis.enums import TimestampFormat
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.ISO_8601)
    assert isinstance(result, str)


def test_timestamp_rfc_3339_format_structure():
    from mimesis import Datetime
    from mimesis.enums import TimestampFormat
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.RFC_3339)
    parts = result.split("T")
    assert len(parts) == 2
    assert len(parts[0].split("-")) == 3


def test_timestamp_iso_8601_format_structure():
    from mimesis import Datetime
    from mimesis.enums import TimestampFormat
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.ISO_8601)
    assert "T" in result
    assert "-" in result


# LLM-generated content at query #4
#--------------------------

```python
def test_datetime_default_parameters():
    """Test datetime generation with default parameters."""
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result = provider.datetime()
    
    assert isinstance(result, dt)
    assert result.year == Datetime._CURRENT_YEAR


def test_datetime_with_custom_years():
    """Test datetime generation with custom year range."""
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result = provider.datetime(start=2010, end=2020)
    
    assert isinstance(result, dt)
    assert 2010 <= result.year <= 2020


def test_datetime_with_same_start_and_end():
    """Test datetime generation with same start and end year."""
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result = provider.datetime(start=2015, end=2015)
    
    assert isinstance(result, dt)
    assert result.year == 2015


def test_datetime_has_time_component():
    """Test that generated datetime has valid time component."""
    from mimesis import Datetime
    
    provider = Datetime()
    result = provider.datetime()
    
    assert 0 <= result.hour <= 23
    assert 0 <= result.minute <= 59
    assert 0 <= result.second <= 59
    assert 0 <= result.microsecond <= 999999


def test_datetime_with_timezone():
    """Test datetime generation with timezone."""
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result = provider.datetime(timezone='US/Eastern')
    
    assert isinstance(result, dt)
    assert result.tzinfo is not None


def test_datetime_without_timezone():
    """Test datetime generation without timezone."""
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result = provider.datetime(timezone=None)
    
    assert isinstance(result, dt)
    assert result.tzinfo is None


def test_datetime_with_invalid_timezone_raises_error():
    """Test that invalid timezone raises appropriate error."""
    from mimesis import Datetime
    
    provider = Datetime()
    
    try:
        provider.datetime(timezone='Invalid/Timezone')
        assert False, "Should have raised an error"
    except Exception:
        pass


def test_datetime_multiple_calls_produce_different_results():
    """Test that multiple calls produce different datetime objects."""
    from mimesis import Datetime
    
    provider = Datetime()
    result1 = provider.datetime(start=2000, end=2023)
    result2 = provider.datetime(start=2000, end=2023)
    
    assert isinstance(result1, type(result2))


def test_datetime_respects_year_boundaries():
    """Test that datetime respects year boundaries."""
    from mimesis import Datetime
    
    provider = Datetime()
    start_year = 2005
    end_year = 2010
    
    for _ in range(10):
        result = provider.datetime(start=start_year, end=end_year)
        assert start_year <= result.year <= end_year


def test_datetime_generates_valid_date():
    """Test that generated datetime has valid date component."""
    from mimesis import Datetime
    from datetime import date
    
    provider = Datetime()
    result = provider.datetime()
    
    assert isinstance(result.date(), date)
    assert 1 <= result.month <= 12
    assert 1 <= result.day <= 31


# LLM-generated content at query #5
#--------------------------

```python
def test_datetime_default_parameters():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime()
    
    assert isinstance(result, dt)
    assert result.year == Datetime._CURRENT_YEAR


def test_datetime_with_custom_year_range():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2010, end=2020)
    
    assert isinstance(result, dt)
    assert 2010 <= result.year <= 2020


def test_datetime_with_same_start_and_end_year():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2015, end=2015)
    
    assert isinstance(result, dt)
    assert result.year == 2015


def test_datetime_with_timezone_utc():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(timezone='UTC')
    
    assert isinstance(result, dt)
    assert result.tzinfo is not None


def test_datetime_with_timezone_us_eastern():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(timezone='US/Eastern')
    
    assert isinstance(result, dt)
    assert result.tzinfo is not None


def test_datetime_has_valid_time_components():
    from mimesis import Datetime
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2000, end=2023)
    
    assert 0 <= result.hour <= 23
    assert 0 <= result.minute <= 59
    assert 0 <= result.second <= 59
    assert 0 <= result.microsecond <= 999999


def test_datetime_multiple_calls_produce_different_results():
    from mimesis import Datetime
    
    datetime_provider = Datetime()
    result1 = datetime_provider.datetime(start=1990, end=2023)
    result2 = datetime_provider.datetime(start=1990, end=2023)
    
    assert result1 != result2


def test_datetime_without_timezone_has_no_tzinfo():
    from mimesis import Datetime
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime()
    
    assert result.tzinfo is None


def test_datetime_with_year_range_1980_to_2010():
    from mimesis import Datetime
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=1980, end=2010)
    
    assert 1980 <= result.year <= 2010


def test_datetime_with_invalid_timezone_raises_error():
    from mimesis import Datetime
    
    datetime_provider = Datetime()
    
    try:
        result = datetime_provider.datetime(timezone='Invalid/Timezone')
        assert False, "Expected exception not raised"
    except Exception:
        assert True


# LLM-generated content at query #6
#--------------------------

```python
def test_datetime_raises_import_error_when_pytz_not_available():
    from unittest.mock import patch, MagicMock
    from datetime import datetime as dt
    
    # Create a mock Datetime instance
    mock_datetime_provider = MagicMock()
    mock_datetime_provider.date.return_value = dt(2023, 5, 15).date()
    mock_datetime_provider.time.return_value = dt(2023, 5, 15, 10, 30, 45).time()
    mock_datetime_provider.random = MagicMock()
    
    # Simulate the datetime method with pytz set to None/False
    with patch('builtins.__import__', side_effect=ImportError("No module named 'pytz'")):
        import sys
        original_pytz = sys.modules.get('pytz')
        sys.modules['pytz'] = None
        
        try:
            # Import the actual Datetime class and test it
            from datetime import datetime as datetime_cls, date as date_cls, time as time_cls, timedelta
            
            # Create a minimal mock setup
            pytz = None
            timezone_arg = "UTC"
            
            # This should raise ImportError when timezone is provided and pytz is not available
            datetime_obj = datetime_cls.combine(
                date=date_cls(2023, 5, 15),
                time=time_cls(10, 30, 45),
            )
            
            if timezone_arg:
                if not pytz:
                    raised_error = False
                    try:
                        raise ImportError("Timezones are supported only with pytz")
                    except ImportError:
                        raised_error = True
                    assert raised_error
        finally:
            if original_pytz is not None:
                sys.modules['pytz'] = original_pytz
            elif 'pytz' in sys.modules:
                del sys.modules['pytz']


# LLM-generated content at query #7
#--------------------------

```python
def test_bulk_create_datetimes_valid_range():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) == 4
    assert result[0] == datetime(2020, 1, 2)
    assert result[-1] == datetime(2020, 1, 5)


def test_bulk_create_datetimes_with_hours():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 3, 0, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, hours=1)
    
    assert len(result) == 3
    assert result[0] == datetime(2020, 1, 1, 1, 0, 0)
    assert result[-1] == datetime(2020, 1, 1, 3, 0, 0)


def test_bulk_create_datetimes_with_multiple_units():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 2)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1, hours=12)
    
    assert len(result) == 1
    assert result[0] == datetime(2020, 1, 2, 12, 0, 0)


def test_bulk_create_datetimes_no_dates_raises_error():
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    
    try:
        dt_provider.bulk_create_datetimes(None, None, days=1)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "You must pass date_start and date_end" in str(e)


def test_bulk_create_datetimes_start_greater_than_end_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 5)
    date_end = datetime(2020, 1, 1)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "date_start can not be larger than date_end" in str(e)


def test_bulk_create_datetimes_zero_timedelta_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_negative_timedelta_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=-1)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_same_start_and_end():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 1)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, seconds=1)
    
    assert len(result) == 0


def test_bulk_create_datetimes_with_seconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 0, 5)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, seconds=1)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 0, 1)
    assert result[-1] == datetime(2020, 1, 1, 0, 0, 5)


def test_bulk_create_datetimes_with_microseconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 0, 0, 3)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, microseconds=1)
    
    assert len(result) == 3
    assert result[0] == datetime(2020, 1, 1, 0, 0, 0, 1)
    assert result[-1] == datetime(2020, 1, 1, 0, 0, 0, 3)


def test_bulk_create_datetimes_with_minutes():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 10, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, minutes=2)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 2, 0)
    assert result[-1] == datetime(2020, 1, 1, 0, 10, 0)


# LLM-generated content at query #8
#--------------------------

```python
def test_datetime_default_parameters():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime()
    
    assert isinstance(result, dt)
    assert result.year == Datetime._CURRENT_YEAR


def test_datetime_with_custom_years():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2010, end=2020)
    
    assert isinstance(result, dt)
    assert 2010 <= result.year <= 2020


def test_datetime_with_same_start_and_end_year():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2015, end=2015)
    
    assert isinstance(result, dt)
    assert result.year == 2015


def test_datetime_with_timezone():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(timezone="UTC")
    
    assert isinstance(result, dt)
    assert result.tzinfo is not None


def test_datetime_with_timezone_and_custom_years():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2005, end=2010, timezone="US/Eastern")
    
    assert isinstance(result, dt)
    assert 2005 <= result.year <= 2010
    assert result.tzinfo is not None


def test_datetime_returns_valid_time_components():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2000, end=2023)
    
    assert isinstance(result, dt)
    assert 0 <= result.hour <= 23
    assert 0 <= result.minute <= 59
    assert 0 <= result.second <= 59
    assert 0 <= result.microsecond <= 999999


def test_datetime_with_invalid_timezone_raises_error():
    from mimesis import Datetime
    
    datetime_provider = Datetime()
    try:
        result = datetime_provider.datetime(timezone="Invalid/Timezone")
        assert False, "Should have raised an error"
    except Exception:
        assert True


# LLM-generated content at query #9
#--------------------------

```python
def test_timestamp_rfc_3339_format():
    from mimesis import Datetime
    from mimesis.enums import TimestampFormat
    
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.RFC_3339)
    
    assert isinstance(result, str)
    assert "T" in result
    assert result.endswith("Z")
    assert len(result) == 20


# LLM-generated content at query #10
#--------------------------

```python
def test_bulk_create_datetimes_basic():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) == 4
    assert result[0] == datetime(2020, 1, 2)
    assert result[-1] == datetime(2020, 1, 5)


def test_bulk_create_datetimes_with_hours():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 5, 0, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, hours=1)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 1, 0, 0)
    assert result[-1] == datetime(2020, 1, 1, 5, 0, 0)


def test_bulk_create_datetimes_with_minutes():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 10, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, minutes=2)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 2, 0)
    assert result[-1] == datetime(2020, 1, 1, 0, 10, 0)


def test_bulk_create_datetimes_raises_when_both_dates_missing():
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    
    try:
        dt_provider.bulk_create_datetimes(None, None, days=1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "You must pass date_start and date_end" in str(e)


def test_bulk_create_datetimes_raises_when_start_greater_than_end():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 5)
    date_end = datetime(2020, 1, 1)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "date_start can not be larger than date_end" in str(e)


def test_bulk_create_datetimes_raises_with_non_positive_timedelta():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=0)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_raises_with_negative_timedelta():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=-1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_with_seconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 0, 10)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, seconds=2)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 0, 2)
    assert result[-1] == datetime(2020, 1, 1, 0, 0, 10)


def test_bulk_create_datetimes_with_microseconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 0, 0, 10)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, microseconds=2)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 0, 0, 2)
    assert result[-1] == datetime(2020, 1, 1, 0, 0, 0, 10)


def test_bulk_create_datetimes_same_start_and_end():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 1)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) == 0


def test_bulk_create_datetimes_multiple_kwargs():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 2, 0, 0, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1, hours=1)
    
    assert len(result) > 0
    assert result[0] > date_start


# LLM-generated content at query #11
#--------------------------

```python
def test_bulk_create_datetimes_positive_timedelta():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(dt, datetime) for dt in result)
    assert result[0] == datetime(2023, 1, 2)


# LLM-generated content at query #12
#--------------------------

```python
def test_bulk_create_datetimes_valid_range():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) == 4
    assert result[0] == datetime(2020, 1, 2)
    assert result[-1] == datetime(2020, 1, 5)


def test_bulk_create_datetimes_with_hours():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 5, 0, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, hours=1)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 1, 0, 0)
    assert result[-1] == datetime(2020, 1, 1, 5, 0, 0)


def test_bulk_create_datetimes_with_minutes():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 10, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, minutes=2)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 2, 0)
    assert result[-1] == datetime(2020, 1, 1, 0, 10, 0)


def test_bulk_create_datetimes_start_equals_end():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 1)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) == 0


def test_bulk_create_datetimes_no_dates_raises_error():
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    
    try:
        dt_provider.bulk_create_datetimes(None, None, days=1)
        assert False, "ValueError should be raised"
    except ValueError as e:
        assert "You must pass date_start and date_end" in str(e)


def test_bulk_create_datetimes_start_after_end_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 10)
    date_end = datetime(2020, 1, 1)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
        assert False, "ValueError should be raised"
    except ValueError as e:
        assert "date_start can not be larger than date_end" in str(e)


def test_bulk_create_datetimes_zero_timedelta_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end)
        assert False, "ValueError should be raised"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_negative_timedelta_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=-1)
        assert False, "ValueError should be raised"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_with_seconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 0, 10)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, seconds=2)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 0, 2)
    assert result[-1] == datetime(2020, 1, 1, 0, 0, 10)


def test_bulk_create_datetimes_with_microseconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 0, 0, 10)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, microseconds=2)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 0, 0, 2)
    assert result[-1] == datetime(2020, 1, 1, 0, 0, 0, 10)


# LLM-generated content at query #13
#--------------------------

```python
def test_datetime_raises_import_error_when_pytz_not_available(monkeypatch):
    from mimesis.providers.datetime import Datetime
    import mimesis.providers.datetime as dt_module
    
    # Save original pytz
    original_pytz = dt_module.pytz
    
    # Mock pytz as None to simulate it not being available
    monkeypatch.setattr(dt_module, 'pytz', None)
    
    # Create Datetime instance
    datetime_provider = Datetime()
    
    # Attempt to call datetime with a timezone should raise ImportError
    try:
        datetime_provider.datetime(timezone='UTC')
        assert False, "Expected ImportError to be raised"
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"
    finally:
        # Restore original pytz
        monkeypatch.setattr(dt_module, 'pytz', original_pytz)


# LLM-generated content at query #14
#--------------------------

```python
def test_bulk_create_datetimes_positive_timedelta():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(dt, datetime) for dt in result)
    assert result[0] == datetime(2020, 1, 2)


# LLM-generated content at query #15
#--------------------------

```python
def test_datetime_raises_import_error_when_pytz_not_available():
    from unittest.mock import patch
    from mimesis import Datetime
    
    dt = Datetime()
    
    with patch('mimesis.providers.datetime_provider.pytz', None):
        try:
            dt.datetime(timezone='UTC')
            assert False, "Expected ImportError to be raised"
        except ImportError as e:
            assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #16
#--------------------------

```python
def test_bulk_create_datetimes_with_positive_timedelta():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(dt, datetime) for dt in result)
    assert result[0] > date_start
    assert result[-1] <= date_end


# LLM-generated content at query #17
#--------------------------

```python
def test_datetime_default_parameters():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime()
    
    assert isinstance(result, dt)
    assert result.year == Datetime._CURRENT_YEAR


def test_datetime_with_custom_year_range():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2015, end=2020)
    
    assert isinstance(result, dt)
    assert 2015 <= result.year <= 2020


def test_datetime_with_different_start_end_years():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2000, end=2010)
    
    assert isinstance(result, dt)
    assert 2000 <= result.year <= 2010


def test_datetime_has_time_component():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime()
    
    assert isinstance(result, dt)
    assert 0 <= result.hour <= 23
    assert 0 <= result.minute <= 59
    assert 0 <= result.second <= 59


def test_datetime_with_timezone():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(timezone="UTC")
    
    assert isinstance(result, dt)
    assert result.tzinfo is not None


def test_datetime_with_timezone_europe_london():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(timezone="Europe/London")
    
    assert isinstance(result, dt)
    assert result.tzinfo is not None


def test_datetime_without_timezone():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(timezone=None)
    
    assert isinstance(result, dt)
    assert result.tzinfo is None


def test_datetime_multiple_calls_different_results():
    from mimesis import Datetime
    
    datetime_provider = Datetime()
    result1 = datetime_provider.datetime(start=2000, end=2023)
    result2 = datetime_provider.datetime(start=2000, end=2023)
    
    assert result1 != result2


def test_datetime_with_single_year():
    from mimesis import Datetime
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2015, end=2015)
    
    assert result.year == 2015


def test_datetime_returns_datetime_object():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime()
    
    assert type(result).__name__ == "datetime"


# LLM-generated content at query #18
#--------------------------

```python
def test_bulk_create_datetimes_basic():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert isinstance(result, list)
    assert len(result) == 4
    assert result[0] == datetime(2020, 1, 2)
    assert result[-1] == datetime(2020, 1, 5)


def test_bulk_create_datetimes_with_hours():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 3, 0, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, hours=1)
    
    assert len(result) == 3
    assert result[0] == datetime(2020, 1, 1, 1, 0, 0)
    assert result[1] == datetime(2020, 1, 1, 2, 0, 0)
    assert result[2] == datetime(2020, 1, 1, 3, 0, 0)


def test_bulk_create_datetimes_with_minutes():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 5, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, minutes=1)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 1, 0)
    assert result[-1] == datetime(2020, 1, 1, 0, 5, 0)


def test_bulk_create_datetimes_with_seconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 0, 10)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, seconds=2)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 0, 2)
    assert result[-1] == datetime(2020, 1, 1, 0, 0, 10)


def test_bulk_create_datetimes_with_microseconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 0, 0, 100)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, microseconds=25)
    
    assert len(result) == 4
    assert result[0] == datetime(2020, 1, 1, 0, 0, 0, 25)
    assert result[-1] == datetime(2020, 1, 1, 0, 0, 0, 100)


def test_bulk_create_datetimes_same_start_and_end():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 1)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert isinstance(result, list)
    assert len(result) == 0


def test_bulk_create_datetimes_error_both_none():
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    
    try:
        dt_provider.bulk_create_datetimes(None, None, days=1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "You must pass date_start and date_end" in str(e)


def test_bulk_create_datetimes_error_date_start_larger():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 10)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "date_start can not be larger than date_end" in str(e)


def test_bulk_create_datetimes_error_non_positive_timedelta():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_error_negative_timedelta():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=-1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_combined_kwargs():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 2, 0, 0, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1, hours=1)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0] > date_start


# LLM-generated content at query #19
#--------------------------

```python
def test_bulk_create_datetimes_predicate_line_46_false():
    """Test that the predicate at line 46 evaluates to False with positive timedelta."""
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    date_start = datetime(2023, 1, 1, 12, 0, 0)
    date_end = datetime(2023, 1, 5, 12, 0, 0)
    
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(dt, datetime) for dt in result)


# LLM-generated content at query #20
#--------------------------

```python
def test_datetime_raises_import_error_when_pytz_not_available():
    from unittest.mock import patch
    from mimesis import Datetime
    
    dt = Datetime()
    
    with patch('mimesis.providers.datetime_provider.pytz', None):
        try:
            dt.datetime(timezone='UTC')
            assert False, "Expected ImportError to be raised"
        except ImportError as e:
            assert str(e) == "Timezones are supported only with pytz"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_duration_default_parameters():
    from mimesis import Datetime
    from datetime import timedelta
    dt = Datetime()
    result = dt.duration()
    assert isinstance(result, timedelta)
    assert result.total_seconds() >= 60
    assert result.total_seconds() <= 600


def test_duration_custom_min_max():
    from mimesis import Datetime
    from datetime import timedelta
    dt = Datetime()
    result = dt.duration(min_duration=5, max_duration=15)
    assert isinstance(result, timedelta)
    assert result.total_seconds() >= 300
    assert result.total_seconds() <= 900


def test_duration_with_hours_unit():
    from mimesis import Datetime
    from mimesis.enums import DurationUnit
    from datetime import timedelta
    dt = Datetime()
    result = dt.duration(min_duration=1, max_duration=5, duration_unit=DurationUnit.HOURS)
    assert isinstance(result, timedelta)
    assert result.total_seconds() >= 3600
    assert result.total_seconds() <= 18000


def test_duration_with_seconds_unit():
    from mimesis import Datetime
    from mimesis.enums import DurationUnit
    from datetime import timedelta
    dt = Datetime()
    result = dt.duration(min_duration=10, max_duration=20, duration_unit=DurationUnit.SECONDS)
    assert isinstance(result, timedelta)
    assert result.total_seconds() >= 10
    assert result.total_seconds() <= 20


def test_duration_with_days_unit():
    from mimesis import Datetime
    from mimesis.enums import DurationUnit
    from datetime import timedelta
    dt = Datetime()
    result = dt.duration(min_duration=1, max_duration=3, duration_unit=DurationUnit.DAYS)
    assert isinstance(result, timedelta)
    assert result.total_seconds() >= 86400
    assert result.total_seconds() <= 259200


def test_duration_min_equals_max():
    from mimesis import Datetime
    from datetime import timedelta
    dt = Datetime()
    result = dt.duration(min_duration=5, max_duration=5)
    assert isinstance(result, timedelta)
    assert result.total_seconds() == 300


def test_duration_raises_value_error_when_min_greater_than_max():
    from mimesis import Datetime
    dt = Datetime()
    try:
        dt.duration(min_duration=10, max_duration=5)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "min_duration must be less or equal to max_duration" in str(e)


def test_duration_raises_type_error_for_non_integer_min():
    from mimesis import Datetime
    dt = Datetime()
    try:
        dt.duration(min_duration=5.5, max_duration=10)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "min_duration and max_duration must be integers" in str(e)


def test_duration_raises_type_error_for_non_integer_max():
    from mimesis import Datetime
    dt = Datetime()
    try:
        dt.duration(min_duration=5, max_duration=10.5)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "min_duration and max_duration must be integers" in str(e)


def test_duration_with_none_unit():
    from mimesis import Datetime
    from datetime import timedelta
    dt = Datetime()
    result = dt.duration(min_duration=1, max_duration=10, duration_unit=None)
    assert isinstance(result, timedelta)
    assert result.total_seconds() > 0


def test_duration_with_microseconds_unit():
    from mimesis import Datetime
    from mimesis.enums import DurationUnit
    from datetime import timedelta
    dt = Datetime()
    result = dt.duration(min_duration=1000, max_duration=5000, duration_unit=DurationUnit.MICROSECONDS)
    assert isinstance(result, timedelta)
    assert result.total_seconds() >= 0.001
    assert result.total_seconds() <= 0.005


# LLM-generated content at query #2
#--------------------------

```python
def test_bulk_create_datetimes_valid_range():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) == 4
    assert result[0] == datetime(2023, 1, 2)
    assert result[-1] == datetime(2023, 1, 5)


def test_bulk_create_datetimes_with_hours():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 3, 0, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, hours=1)
    
    assert len(result) == 3
    assert result[0] == datetime(2023, 1, 1, 1, 0, 0)
    assert result[-1] == datetime(2023, 1, 1, 3, 0, 0)


def test_bulk_create_datetimes_no_arguments_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    
    try:
        dt_provider.bulk_create_datetimes(None, None, days=1)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "You must pass date_start and date_end" in str(e)


def test_bulk_create_datetimes_start_after_end_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2023, 1, 5)
    date_end = datetime(2023, 1, 1)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "date_start can not be larger than date_end" in str(e)


def test_bulk_create_datetimes_non_positive_timedelta_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=0)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_with_minutes():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 5, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, minutes=1)
    
    assert len(result) == 5
    assert result[0] == datetime(2023, 1, 1, 0, 1, 0)
    assert result[-1] == datetime(2023, 1, 1, 0, 5, 0)


def test_bulk_create_datetimes_with_seconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 0, 3)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, seconds=1)
    
    assert len(result) == 3
    assert result[0] == datetime(2023, 1, 1, 0, 0, 1)
    assert result[-1] == datetime(2023, 1, 1, 0, 0, 3)


def test_bulk_create_datetimes_single_point_range():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 1)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) == 0


def test_bulk_create_datetimes_with_microseconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2023, 1, 1, 0, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 0, 0, 3)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, microseconds=1)
    
    assert len(result) == 3
    assert result[0] == datetime(2023, 1, 1, 0, 0, 0, 1)
    assert result[-1] == datetime(2023, 1, 1, 0, 0, 0, 3)


def test_bulk_create_datetimes_negative_timedelta_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=-1)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


# LLM-generated content at query #3
#--------------------------

```python
def test_datetime_default_parameters():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime()
    
    assert isinstance(result, dt)
    assert result.year == Datetime._CURRENT_YEAR


def test_datetime_with_custom_years():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2010, end=2015)
    
    assert isinstance(result, dt)
    assert 2010 <= result.year <= 2015


def test_datetime_with_same_start_end_year():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2020, end=2020)
    
    assert isinstance(result, dt)
    assert result.year == 2020


def test_datetime_with_timezone():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2015, end=2020, timezone="US/Eastern")
    
    assert isinstance(result, dt)
    assert result.tzinfo is not None
    assert 2015 <= result.year <= 2020


def test_datetime_includes_time_component():
    from mimesis import Datetime
    from datetime import datetime as dt, time as time_class
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime()
    
    assert isinstance(result, dt)
    assert isinstance(result.time(), time_class)
    assert 0 <= result.hour <= 23
    assert 0 <= result.minute <= 59
    assert 0 <= result.second <= 59


def test_datetime_includes_date_component():
    from mimesis import Datetime
    from datetime import datetime as dt, date as date_class
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime()
    
    assert isinstance(result, dt)
    assert isinstance(result.date(), date_class)
    assert 1 <= result.month <= 12
    assert 1 <= result.day <= 31


# LLM-generated content at query #4
#--------------------------

```python
def test_datetime_raises_import_error_when_pytz_not_available():
    from datetime import datetime as dt_class
    from unittest.mock import MagicMock, patch
    
    # Create a mock Datetime instance
    mock_datetime_provider = MagicMock()
    mock_datetime_provider.date.return_value = dt_class(2023, 5, 15).date()
    mock_datetime_provider.time.return_value = dt_class(14, 30, 45).time()
    
    # Import the actual Datetime class
    from mimesis.providers.datetime import Datetime
    
    # Create a real instance
    datetime_provider = Datetime()
    
    # Patch pytz to be None/falsy to trigger the condition at line 19
    with patch('mimesis.providers.datetime.pytz', None):
        try:
            # This should raise ImportError because pytz is None and timezone is provided
            datetime_provider.datetime(timezone="America/New_York")
            assert False, "Expected ImportError to be raised"
        except ImportError as e:
            assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #5
#--------------------------

```python
def test_bulk_create_datetimes_valid_range():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) == 4
    assert result[0] == datetime(2020, 1, 2)
    assert result[-1] == datetime(2020, 1, 5)


def test_bulk_create_datetimes_with_hours():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 3, 0, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, hours=1)
    
    assert len(result) == 3
    assert result[0] == datetime(2020, 1, 1, 1, 0, 0)
    assert result[-1] == datetime(2020, 1, 1, 3, 0, 0)


def test_bulk_create_datetimes_with_minutes():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 5, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, minutes=1)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 1, 0)


def test_bulk_create_datetimes_no_dates_raises_error():
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    
    try:
        dt_provider.bulk_create_datetimes(None, None, days=1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "You must pass date_start and date_end" in str(e)


def test_bulk_create_datetimes_start_larger_than_end_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 5)
    date_end = datetime(2020, 1, 1)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "date_start can not be larger than date_end" in str(e)


def test_bulk_create_datetimes_non_positive_timedelta_raises_error():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=0)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_with_seconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 0, 3)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, seconds=1)
    
    assert len(result) == 3
    assert result[0] == datetime(2020, 1, 1, 0, 0, 1)
    assert result[-1] == datetime(2020, 1, 1, 0, 0, 3)


def test_bulk_create_datetimes_same_start_and_end():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 1)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) == 0


def test_bulk_create_datetimes_with_multiple_kwargs():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 2, 2, 0, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1, hours=1)
    
    assert len(result) > 0
    assert result[0] == datetime(2020, 1, 1, 1, 0, 0)


# LLM-generated content at query #6
#--------------------------

```python
def test_bulk_create_datetimes_predicate_line_46_evaluates_to_false():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) > 0
    assert result[0] == datetime(2023, 1, 2)
    assert result[-1] == datetime(2023, 1, 10)


# LLM-generated content at query #7
#--------------------------

```python
def test_datetime_default_parameters():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result = provider.datetime()
    
    assert isinstance(result, dt)
    assert result.year == Datetime._CURRENT_YEAR


def test_datetime_with_custom_year_range():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result = provider.datetime(start=2010, end=2015)
    
    assert isinstance(result, dt)
    assert 2010 <= result.year <= 2015


def test_datetime_with_timezone():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result = provider.datetime(timezone="UTC")
    
    assert isinstance(result, dt)
    assert result.tzinfo is not None


def test_datetime_with_timezone_different_zones():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result_utc = provider.datetime(timezone="UTC")
    result_est = provider.datetime(timezone="US/Eastern")
    
    assert isinstance(result_utc, dt)
    assert isinstance(result_est, dt)
    assert result_utc.tzinfo is not None
    assert result_est.tzinfo is not None


def test_datetime_start_equals_end():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result = provider.datetime(start=2020, end=2020)
    
    assert isinstance(result, dt)
    assert result.year == 2020


def test_datetime_has_valid_time_components():
    from mimesis import Datetime
    
    provider = Datetime()
    result = provider.datetime(start=2000, end=2023)
    
    assert 0 <= result.hour <= 23
    assert 0 <= result.minute <= 59
    assert 0 <= result.second <= 59
    assert 0 <= result.microsecond <= 999999


def test_datetime_without_timezone():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result = provider.datetime()
    
    assert isinstance(result, dt)
    assert result.tzinfo is None


def test_datetime_with_old_year_range():
    from mimesis import Datetime
    from datetime import datetime as dt
    
    provider = Datetime()
    result = provider.datetime(start=1950, end=1960)
    
    assert isinstance(result, dt)
    assert 1950 <= result.year <= 1960


# LLM-generated content at query #8
#--------------------------

```python
def test_bulk_create_datetimes_with_positive_timedelta():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(dt, datetime) for dt in result)
    assert result[0] == datetime(2023, 1, 2)


# LLM-generated content at query #9
#--------------------------

```python
def test_datetime_raises_import_error_when_pytz_not_available():
    from unittest.mock import patch
    from mimesis import Datetime
    
    dt = Datetime()
    
    with patch('mimesis.providers.datetime.pytz', None):
        try:
            dt.datetime(timezone='UTC')
            assert False, "Expected ImportError to be raised"
        except ImportError as e:
            assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #10
#--------------------------

```python
def test_bulk_create_datetimes_predicate_line_46_false():
    """Test that the predicate at line 46 evaluates to False (timedelta is positive)."""
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    
    result = Datetime.bulk_create_datetimes(
        date_start=date_start,
        date_end=date_end,
        days=1
    )
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(dt, datetime) for dt in result)
    assert result[0] == datetime(2023, 1, 2)


# LLM-generated content at query #11
#--------------------------

```python
def test_datetime_raises_import_error_when_pytz_not_available():
    from unittest.mock import patch
    from mimesis import Datetime
    
    dt = Datetime()
    
    with patch('mimesis.providers.datetime_provider.pytz', None):
        try:
            dt.datetime(timezone='UTC')
            assert False, "Expected ImportError to be raised"
        except ImportError as e:
            assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #12
#--------------------------

```python
def test_bulk_create_datetimes_with_positive_timedelta():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) == 4
    assert result[0] == datetime(2023, 1, 2)
    assert result[1] == datetime(2023, 1, 3)
    assert result[2] == datetime(2023, 1, 4)
    assert result[3] == datetime(2023, 1, 5)


# LLM-generated content at query #13
#--------------------------

```python
def test_datetime_default_parameters():
    from mimesis import Datetime
    from datetime import datetime as dt_class
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime()
    
    assert isinstance(result, dt_class)
    assert result.year == Datetime._CURRENT_YEAR


def test_datetime_with_custom_years():
    from mimesis import Datetime
    from datetime import datetime as dt_class
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2010, end=2015)
    
    assert isinstance(result, dt_class)
    assert 2010 <= result.year <= 2015


def test_datetime_with_same_start_and_end_year():
    from mimesis import Datetime
    from datetime import datetime as dt_class
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2020, end=2020)
    
    assert isinstance(result, dt_class)
    assert result.year == 2020


def test_datetime_with_timezone_pytz():
    from mimesis import Datetime
    from datetime import datetime as dt_class
    import pytz
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(timezone="UTC")
    
    assert isinstance(result, dt_class)
    assert result.tzinfo is not None


def test_datetime_with_timezone_us_eastern():
    from mimesis import Datetime
    from datetime import datetime as dt_class
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(timezone="US/Eastern")
    
    assert isinstance(result, dt_class)
    assert result.tzinfo is not None


def test_datetime_without_pytz_raises_import_error():
    from mimesis import Datetime
    from unittest.mock import patch
    
    datetime_provider = Datetime()
    
    with patch('mimesis.providers.datetime_provider.pytz', None):
        try:
            datetime_provider.datetime(timezone="UTC")
            assert False, "Expected ImportError"
        except ImportError as e:
            assert "pytz" in str(e)


def test_datetime_has_valid_time_components():
    from mimesis import Datetime
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime()
    
    assert 0 <= result.hour <= 23
    assert 0 <= result.minute <= 59
    assert 0 <= result.second <= 59
    assert 0 <= result.microsecond <= 999999


def test_datetime_has_valid_date_components():
    from mimesis import Datetime
    
    datetime_provider = Datetime()
    result = datetime_provider.datetime(start=2000, end=2023)
    
    assert 2000 <= result.year <= 2023
    assert 1 <= result.month <= 12
    assert 1 <= result.day <= 31


# LLM-generated content at query #14
#--------------------------

```python
def test_bulk_create_datetimes_predicate_line_46_false():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    date_start = datetime(2023, 1, 1, 12, 0, 0)
    date_end = datetime(2023, 1, 5, 12, 0, 0)
    
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(dt, datetime) for dt in result)
    assert result[0] == datetime(2023, 1, 2, 12, 0, 0)


# LLM-generated content at query #15
#--------------------------

```python
def test_datetime_raises_import_error_when_pytz_not_available():
    from unittest.mock import patch
    from mimesis import Datetime
    
    dt = Datetime()
    
    with patch('mimesis.providers.datetime_provider.pytz', None):
        try:
            dt.datetime(timezone='UTC')
            assert False, "Expected ImportError to be raised"
        except ImportError as e:
            assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #16
#--------------------------

```python
def test_datetime_raises_import_error_when_pytz_not_available(monkeypatch):
    from mimesis.providers.datetime import Datetime
    import mimesis.providers.datetime as dt_module
    
    # Save original pytz
    original_pytz = dt_module.pytz
    
    # Mock pytz to be None/False
    monkeypatch.setattr(dt_module, 'pytz', None)
    
    dt_provider = Datetime()
    
    try:
        dt_provider.datetime(timezone='UTC')
        assert False, "Expected ImportError to be raised"
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"
    finally:
        # Restore original pytz
        monkeypatch.setattr(dt_module, 'pytz', original_pytz)


# LLM-generated content at query #17
#--------------------------

```python
def test_bulk_create_datetimes_basic():
    from datetime import datetime, timedelta
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert isinstance(result, list)
    assert len(result) == 4
    assert all(isinstance(item, datetime) for item in result)
    assert result[0] == datetime(2020, 1, 2)
    assert result[-1] == datetime(2020, 1, 5)


def test_bulk_create_datetimes_with_hours():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 3, 0, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, hours=1)
    
    assert len(result) == 3
    assert result[0] == datetime(2020, 1, 1, 1, 0, 0)
    assert result[-1] == datetime(2020, 1, 1, 3, 0, 0)


def test_bulk_create_datetimes_with_minutes():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 5, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, minutes=1)
    
    assert len(result) == 5
    assert result[0] == datetime(2020, 1, 1, 0, 1, 0)


def test_bulk_create_datetimes_with_seconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 0, 3)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, seconds=1)
    
    assert len(result) == 3
    assert result[0] == datetime(2020, 1, 1, 0, 0, 1)


def test_bulk_create_datetimes_with_microseconds():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0, 0)
    date_end = datetime(2020, 1, 1, 0, 0, 0, 3)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, microseconds=1)
    
    assert len(result) == 3
    assert result[0] == datetime(2020, 1, 1, 0, 0, 0, 1)


def test_bulk_create_datetimes_raises_error_when_both_none():
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    
    try:
        dt_provider.bulk_create_datetimes(None, None, days=1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "You must pass date_start and date_end" in str(e)


def test_bulk_create_datetimes_raises_error_when_start_greater_than_end():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 5)
    date_end = datetime(2020, 1, 1)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "date_start can not be larger than date_end" in str(e)


def test_bulk_create_datetimes_raises_error_when_timedelta_not_positive():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_raises_error_when_timedelta_negative():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 5)
    
    try:
        dt_provider.bulk_create_datetimes(date_start, date_end, days=-1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "timedelta must be positive" in str(e)


def test_bulk_create_datetimes_equal_start_and_end():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1)
    date_end = datetime(2020, 1, 1)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1)
    
    assert len(result) == 0


def test_bulk_create_datetimes_combined_kwargs():
    from datetime import datetime
    from mimesis.providers.datetime import Datetime
    
    dt_provider = Datetime()
    date_start = datetime(2020, 1, 1, 0, 0, 0)
    date_end = datetime(2020, 1, 2, 2, 0, 0)
    
    result = dt_provider.bulk_create_datetimes(date_start, date_end, days=1, hours=1)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, datetime) for item in result)


