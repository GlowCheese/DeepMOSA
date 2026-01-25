# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.locales as module_0
import mimesis.exceptions as module_1
import builtins as module_2
import mimesis.enums as module_3

def test_case_0():
    var_0 = 'G,\r`,NI3"B|FGYP^]c'
    with pytest.raises(module_1.LocaleError):
        module_0.validate_locale(var_0)

def test_case_1():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_2.Exception(*var_1)
    with pytest.raises(module_1.LocaleError):
        module_0.validate_locale(var_2)

def test_case_2():
    var_0 = module_3.Locale.AR_JO
    var_1 = module_0.validate_locale(var_0)
    assert var_1 == module_3.Locale.AR_JO
    assert f'{type(module_3.Locale.values).__module__}.{type(module_3.Locale.values).__qualname__}' == 'builtins.method'
    assert module_3.Locale.AR_AE == module_3.Locale.AR_AE
    assert module_3.Locale.AR_DZ == module_3.Locale.AR_DZ
    assert module_3.Locale.AR_EG == module_3.Locale.AR_EG
    assert module_3.Locale.AR_JO == module_3.Locale.AR_JO
    assert module_3.Locale.AR_OM == module_3.Locale.AR_OM
    assert module_3.Locale.AR_KW == module_3.Locale.AR_KW
    assert module_3.Locale.AR_MA == module_3.Locale.AR_MA
    assert module_3.Locale.AR_QA == module_3.Locale.AR_QA
    assert module_3.Locale.AR_SA == module_3.Locale.AR_SA
    assert module_3.Locale.AR_SY == module_3.Locale.AR_SY
    assert module_3.Locale.AR_TN == module_3.Locale.AR_TN
    assert module_3.Locale.AR_YE == module_3.Locale.AR_YE
    assert module_3.Locale.AZ == module_3.Locale.AZ
    assert module_3.Locale.CS == module_3.Locale.CS
    assert module_3.Locale.DA == module_3.Locale.DA
    assert module_3.Locale.DE == module_3.Locale.DE
    assert module_3.Locale.DE_AT == module_3.Locale.DE_AT
    assert module_3.Locale.DE_CH == module_3.Locale.DE_CH
    assert module_3.Locale.EL == module_3.Locale.EL
    assert module_3.Locale.EN == module_3.Locale.EN
    assert module_3.Locale.EN_AU == module_3.Locale.EN_AU
    assert module_3.Locale.EN_CA == module_3.Locale.EN_CA
    assert module_3.Locale.EN_GB == module_3.Locale.EN_GB
    assert module_3.Locale.ES == module_3.Locale.ES
    assert module_3.Locale.ES_MX == module_3.Locale.ES_MX
    assert module_3.Locale.ET == module_3.Locale.ET
    assert module_3.Locale.FA == module_3.Locale.FA
    assert module_3.Locale.FI == module_3.Locale.FI
    assert module_3.Locale.FR == module_3.Locale.FR
    assert module_3.Locale.HU == module_3.Locale.HU
    assert module_3.Locale.HR == module_3.Locale.HR
    assert module_3.Locale.IS == module_3.Locale.IS
    assert module_3.Locale.IT == module_3.Locale.IT
    assert module_3.Locale.JA == module_3.Locale.JA
    assert module_3.Locale.KK == module_3.Locale.KK
    assert module_3.Locale.KO == module_3.Locale.KO
    assert module_3.Locale.NL == module_3.Locale.NL
    assert module_3.Locale.NL_BE == module_3.Locale.NL_BE
    assert module_3.Locale.NO == module_3.Locale.NO
    assert module_3.Locale.PL == module_3.Locale.PL
    assert module_3.Locale.PT == module_3.Locale.PT
    assert module_3.Locale.PT_BR == module_3.Locale.PT_BR
    assert module_3.Locale.RU == module_3.Locale.RU
    assert module_3.Locale.SK == module_3.Locale.SK
    assert module_3.Locale.SV == module_3.Locale.SV
    assert module_3.Locale.TR == module_3.Locale.TR
    assert module_3.Locale.UK == module_3.Locale.UK
    assert module_3.Locale.ZH == module_3.Locale.ZH
    assert module_3.Locale.DEFAULT == module_3.Locale.EN
    var_2 = None
    with pytest.raises(module_1.LocaleError):
        module_0.validate_locale(var_2)