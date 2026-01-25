# Check out: https://github.com/GlowCheese/deepmosa
import platform as module_1

import arrow.locales as module_2
import cookiecutter.extensions as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.JsonifyExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.RandomStringExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_1.node()
    assert var_0 == '269b03c4c612'
    module_0.SlugifyExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.UUIDExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_2.MarathiLocale()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'arrow.locales.MarathiLocale'
    assert module_2.MarathiLocale.names == ['mr']
    assert module_2.MarathiLocale.past == '{0} आधी'
    assert module_2.MarathiLocale.future == '{0} नंतर'
    assert module_2.MarathiLocale.timeframes == {'now': 'सद्य', 'second': 'एक सेकंद', 'seconds': '{0} सेकंद', 'minute': 'एक मिनिट ', 'minutes': '{0} मिनिट ', 'hour': 'एक तास', 'hours': '{0} तास', 'day': 'एक दिवस', 'days': '{0} दिवस', 'month': 'एक महिना ', 'months': '{0} महिने ', 'year': 'एक वर्ष ', 'years': '{0} वर्ष '}
    assert module_2.MarathiLocale.meridians == {'am': 'सकाळ', 'pm': 'संध्याकाळ', 'AM': 'सकाळ', 'PM': 'संध्याकाळ'}
    assert module_2.MarathiLocale.month_names == ['', 'जानेवारी', 'फेब्रुवारी', 'मार्च', 'एप्रिल', 'मे', 'जून', 'जुलै', 'अॉगस्ट', 'सप्टेंबर', 'अॉक्टोबर', 'नोव्हेंबर', 'डिसेंबर']
    assert module_2.MarathiLocale.month_abbreviations == ['', 'जान', 'फेब्रु', 'मार्च', 'एप्रि', 'मे', 'जून', 'जुलै', 'अॉग', 'सप्टें', 'अॉक्टो', 'नोव्हें', 'डिसें']
    assert module_2.MarathiLocale.day_names == ['', 'सोमवार', 'मंगळवार', 'बुधवार', 'गुरुवार', 'शुक्रवार', 'शनिवार', 'रविवार']
    assert module_2.MarathiLocale.day_abbreviations == ['', 'सोम', 'मंगळ', 'बुध', 'गुरु', 'शुक्र', 'शनि', 'रवि']
    module_0.TimeExtension(var_0)