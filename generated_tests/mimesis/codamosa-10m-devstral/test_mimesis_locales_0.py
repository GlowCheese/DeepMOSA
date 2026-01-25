# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.enums as module_0
import mimesis.locales as module_1
import mimesis.exceptions as module_2

def test_case_0():
    var_0 = module_0.Locale.EN_AU
    var_1 = None
    var_2 = module_1.validate_locale(var_0)
    assert var_2 == module_0.Locale.EN_AU
    assert f'{type(module_0.Locale.values).__module__}.{type(module_0.Locale.values).__qualname__}' == 'builtins.method'
    assert module_0.Locale.AR_AE == module_0.Locale.AR_AE
    assert module_0.Locale.AR_DZ == module_0.Locale.AR_DZ
    assert module_0.Locale.AR_EG == module_0.Locale.AR_EG
    assert module_0.Locale.AR_JO == module_0.Locale.AR_JO
    assert module_0.Locale.AR_OM == module_0.Locale.AR_OM
    assert module_0.Locale.AR_KW == module_0.Locale.AR_KW
    assert module_0.Locale.AR_MA == module_0.Locale.AR_MA
    assert module_0.Locale.AR_QA == module_0.Locale.AR_QA
    assert module_0.Locale.AR_SA == module_0.Locale.AR_SA
    assert module_0.Locale.AR_SY == module_0.Locale.AR_SY
    assert module_0.Locale.AR_TN == module_0.Locale.AR_TN
    assert module_0.Locale.AR_YE == module_0.Locale.AR_YE
    assert module_0.Locale.AZ == module_0.Locale.AZ
    assert module_0.Locale.CS == module_0.Locale.CS
    assert module_0.Locale.DA == module_0.Locale.DA
    assert module_0.Locale.DE == module_0.Locale.DE
    assert module_0.Locale.DE_AT == module_0.Locale.DE_AT
    assert module_0.Locale.DE_CH == module_0.Locale.DE_CH
    assert module_0.Locale.EL == module_0.Locale.EL
    assert module_0.Locale.EN == module_0.Locale.EN
    assert module_0.Locale.EN_AU == module_0.Locale.EN_AU
    assert module_0.Locale.EN_CA == module_0.Locale.EN_CA
    assert module_0.Locale.EN_GB == module_0.Locale.EN_GB
    assert module_0.Locale.ES == module_0.Locale.ES
    assert module_0.Locale.ES_MX == module_0.Locale.ES_MX
    assert module_0.Locale.ET == module_0.Locale.ET
    assert module_0.Locale.FA == module_0.Locale.FA
    assert module_0.Locale.FI == module_0.Locale.FI
    assert module_0.Locale.FR == module_0.Locale.FR
    assert module_0.Locale.HU == module_0.Locale.HU
    assert module_0.Locale.HR == module_0.Locale.HR
    assert module_0.Locale.IS == module_0.Locale.IS
    assert module_0.Locale.IT == module_0.Locale.IT
    assert module_0.Locale.JA == module_0.Locale.JA
    assert module_0.Locale.KK == module_0.Locale.KK
    assert module_0.Locale.KO == module_0.Locale.KO
    assert module_0.Locale.NL == module_0.Locale.NL
    assert module_0.Locale.NL_BE == module_0.Locale.NL_BE
    assert module_0.Locale.NO == module_0.Locale.NO
    assert module_0.Locale.PL == module_0.Locale.PL
    assert module_0.Locale.PT == module_0.Locale.PT
    assert module_0.Locale.PT_BR == module_0.Locale.PT_BR
    assert module_0.Locale.RU == module_0.Locale.RU
    assert module_0.Locale.SK == module_0.Locale.SK
    assert module_0.Locale.SV == module_0.Locale.SV
    assert module_0.Locale.TR == module_0.Locale.TR
    assert module_0.Locale.UK == module_0.Locale.UK
    assert module_0.Locale.ZH == module_0.Locale.ZH
    assert module_0.Locale.DEFAULT == module_0.Locale.EN
    with pytest.raises(module_2.LocaleError):
        module_1.validate_locale(var_1)

def test_case_1():
    var_0 = None
    with pytest.raises(module_2.LocaleError):
        module_1.validate_locale(var_0)

def test_case_2():
    var_0 = 'g$'
    with pytest.raises(module_2.LocaleError):
        module_1.validate_locale(var_0)