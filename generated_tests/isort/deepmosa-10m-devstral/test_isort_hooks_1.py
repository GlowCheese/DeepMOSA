# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.hooks as module_0
import http.cookiejar as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    module_0.git_hook(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    module_0.git_hook()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = True
    module_0.git_hook(directories=var_0)

def test_case_3():
    var_0 = 'echo'
    var_1 = module_1.reach(var_0)
    assert var_1 == 'echo'
    assert module_1.debug is False
    assert module_1.logger is None
    assert module_1.HTTPONLY_ATTR == 'HTTPOnly'
    assert module_1.HTTPONLY_PREFIX == '#HttpOnly_'
    assert module_1.DEFAULT_HTTP_PORT == '80'
    assert f'{type(module_1.NETSCAPE_MAGIC_RGX).__module__}.{type(module_1.NETSCAPE_MAGIC_RGX).__qualname__}' == 're.Pattern'
    assert module_1.MISSING_FILENAME_TEXT == 'a filename was not supplied (nor was the CookieJar instance initialised with one)'
    assert module_1.NETSCAPE_HEADER_TEXT == '# Netscape HTTP Cookie File\n# http://curl.haxx.se/rfc/cookie_spec.html\n# This is a generated file!  Do not edit.\n\n'
    assert module_1.EPOCH_YEAR == 1970
    assert module_1.DAYS == ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    assert module_1.MONTHS == ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    assert module_1.MONTHS_LOWER == ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    assert module_1.month == 'Dec'
    assert module_1.UTC_ZONES == {'GMT': None, 'UTC': None, 'UT': None, 'Z': None}
    assert f'{type(module_1.TIMEZONE_RE).__module__}.{type(module_1.TIMEZONE_RE).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.STRICT_DATE_RE).__module__}.{type(module_1.STRICT_DATE_RE).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.WEEKDAY_RE).__module__}.{type(module_1.WEEKDAY_RE).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.LOOSE_HTTP_DATE_RE).__module__}.{type(module_1.LOOSE_HTTP_DATE_RE).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.ISO_DATE_RE).__module__}.{type(module_1.ISO_DATE_RE).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.HEADER_TOKEN_RE).__module__}.{type(module_1.HEADER_TOKEN_RE).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.HEADER_QUOTED_VALUE_RE).__module__}.{type(module_1.HEADER_QUOTED_VALUE_RE).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.HEADER_VALUE_RE).__module__}.{type(module_1.HEADER_VALUE_RE).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.HEADER_ESCAPE_RE).__module__}.{type(module_1.HEADER_ESCAPE_RE).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.HEADER_JOIN_ESCAPE_RE).__module__}.{type(module_1.HEADER_JOIN_ESCAPE_RE).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_RE).__module__}.{type(module_1.IPV4_RE).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.cut_port_re).__module__}.{type(module_1.cut_port_re).__qualname__}' == 're.Pattern'
    assert module_1.HTTP_PATH_SAFE == "%/;:@&=+$,!~*'()"
    assert f'{type(module_1.ESCAPED_CHAR_RE).__module__}.{type(module_1.ESCAPED_CHAR_RE).__qualname__}' == 're.Pattern'
    var_2 = module_0.get_lines(var_1)