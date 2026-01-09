# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.enums as module_2
import mimesis.exceptions as module_1
import mimesis.locales as module_0
import pytest


def test_case_0():
    var_0 = ')3=T?FOmF;\r4'
    with pytest.raises(module_1.LocaleError):
        module_0.validate_locale(var_0)

def test_case_1():
    var_0 = None
    with pytest.raises(module_1.LocaleError):
        module_0.validate_locale(var_0)

def test_case_2():
    var_0 = module_2.Locale.SV
    var_1 = module_0.validate_locale(var_0)
    assert var_1 == module_2.Locale.SV
    assert f'{type(module_2.Locale.values).__module__}.{type(module_2.Locale.values).__qualname__}' == 'builtins.method'
    assert module_2.Locale.AR_AE == module_2.Locale.AR_AE
    assert module_2.Locale.AR_DZ == module_2.Locale.AR_DZ
    assert module_2.Locale.AR_EG == module_2.Locale.AR_EG
    assert module_2.Locale.AR_JO == module_2.Locale.AR_JO
    assert module_2.Locale.AR_OM == module_2.Locale.AR_OM
    assert module_2.Locale.AR_KW == module_2.Locale.AR_KW
    assert module_2.Locale.AR_MA == module_2.Locale.AR_MA
    assert module_2.Locale.AR_QA == module_2.Locale.AR_QA
    assert module_2.Locale.AR_SA == module_2.Locale.AR_SA
    assert module_2.Locale.AR_SY == module_2.Locale.AR_SY
    assert module_2.Locale.AR_TN == module_2.Locale.AR_TN
    assert module_2.Locale.AR_YE == module_2.Locale.AR_YE
    assert module_2.Locale.AZ == module_2.Locale.AZ
    assert module_2.Locale.CS == module_2.Locale.CS
    assert module_2.Locale.DA == module_2.Locale.DA
    assert module_2.Locale.DE == module_2.Locale.DE
    assert module_2.Locale.DE_AT == module_2.Locale.DE_AT
    assert module_2.Locale.DE_CH == module_2.Locale.DE_CH
    assert module_2.Locale.EL == module_2.Locale.EL
    assert module_2.Locale.EN == module_2.Locale.EN
    assert module_2.Locale.EN_AU == module_2.Locale.EN_AU
    assert module_2.Locale.EN_CA == module_2.Locale.EN_CA
    assert module_2.Locale.EN_GB == module_2.Locale.EN_GB
    assert module_2.Locale.ES == module_2.Locale.ES
    assert module_2.Locale.ES_MX == module_2.Locale.ES_MX
    assert module_2.Locale.ET == module_2.Locale.ET
    assert module_2.Locale.FA == module_2.Locale.FA
    assert module_2.Locale.FI == module_2.Locale.FI
    assert module_2.Locale.FR == module_2.Locale.FR
    assert module_2.Locale.HU == module_2.Locale.HU
    assert module_2.Locale.HR == module_2.Locale.HR
    assert module_2.Locale.IS == module_2.Locale.IS
    assert module_2.Locale.IT == module_2.Locale.IT
    assert module_2.Locale.JA == module_2.Locale.JA
    assert module_2.Locale.KK == module_2.Locale.KK
    assert module_2.Locale.KO == module_2.Locale.KO
    assert module_2.Locale.NL == module_2.Locale.NL
    assert module_2.Locale.NL_BE == module_2.Locale.NL_BE
    assert module_2.Locale.NO == module_2.Locale.NO
    assert module_2.Locale.PL == module_2.Locale.PL
    assert module_2.Locale.PT == module_2.Locale.PT
    assert module_2.Locale.PT_BR == module_2.Locale.PT_BR
    assert module_2.Locale.RU == module_2.Locale.RU
    assert module_2.Locale.SK == module_2.Locale.SK
    assert module_2.Locale.SV == module_2.Locale.SV
    assert module_2.Locale.TR == module_2.Locale.TR
    assert module_2.Locale.UK == module_2.Locale.UK
    assert module_2.Locale.ZH == module_2.Locale.ZH
    assert module_2.Locale.DEFAULT == module_2.Locale.EN