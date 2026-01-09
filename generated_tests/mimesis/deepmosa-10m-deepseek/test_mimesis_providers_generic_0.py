# Check out: https://github.com/GlowCheese/deepmosa
import ast as module_3

import mimesis.enums as module_1
import mimesis.providers.base as module_2
import mimesis.providers.generic as module_0
import pytest


def test_case_0():
    var_0 = module_0.Generic()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == module_1.Locale.EN
    assert f'{type(var_0.binaryfile).__module__}.{type(var_0.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_0.code).__module__}.{type(var_0.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_0.cryptographic).__module__}.{type(var_0.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_0.development).__module__}.{type(var_0.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_0.file).__module__}.{type(var_0.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_0.hardware).__module__}.{type(var_0.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_0.internet).__module__}.{type(var_0.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_0.numeric).__module__}.{type(var_0.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.path).__module__}.{type(var_0.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_0.payment).__module__}.{type(var_0.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.science).__module__}.{type(var_0.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_0.transport).__module__}.{type(var_0.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'

def test_case_1():
    var_0 = module_0.Generic()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == module_1.Locale.EN
    assert f'{type(var_0.binaryfile).__module__}.{type(var_0.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_0.code).__module__}.{type(var_0.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_0.cryptographic).__module__}.{type(var_0.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_0.development).__module__}.{type(var_0.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_0.file).__module__}.{type(var_0.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_0.hardware).__module__}.{type(var_0.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_0.internet).__module__}.{type(var_0.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_0.numeric).__module__}.{type(var_0.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.path).__module__}.{type(var_0.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_0.payment).__module__}.{type(var_0.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.science).__module__}.{type(var_0.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_0.transport).__module__}.{type(var_0.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.add_providers()

def test_case_2():
    var_0 = None
    var_1 = module_0.Generic(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale is None
    assert f'{type(var_1.binaryfile).__module__}.{type(var_1.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_1.code).__module__}.{type(var_1.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_1.cryptographic).__module__}.{type(var_1.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_1.development).__module__}.{type(var_1.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_1.file).__module__}.{type(var_1.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_1.hardware).__module__}.{type(var_1.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_1.internet).__module__}.{type(var_1.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_1.numeric).__module__}.{type(var_1.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_1.path).__module__}.{type(var_1.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_1.payment).__module__}.{type(var_1.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_1.science).__module__}.{type(var_1.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_1.transport).__module__}.{type(var_1.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = var_1.__str__()
    assert var_2 == 'Generic <None>'

def test_case_3():
    var_0 = module_0.Generic()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == module_1.Locale.EN
    assert f'{type(var_0.binaryfile).__module__}.{type(var_0.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_0.code).__module__}.{type(var_0.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_0.cryptographic).__module__}.{type(var_0.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_0.development).__module__}.{type(var_0.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_0.file).__module__}.{type(var_0.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_0.hardware).__module__}.{type(var_0.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_0.internet).__module__}.{type(var_0.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_0.numeric).__module__}.{type(var_0.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.path).__module__}.{type(var_0.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_0.payment).__module__}.{type(var_0.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.science).__module__}.{type(var_0.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_0.transport).__module__}.{type(var_0.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.reseed()
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.Generic(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale is None
    assert f'{type(var_1.binaryfile).__module__}.{type(var_1.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_1.code).__module__}.{type(var_1.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_1.cryptographic).__module__}.{type(var_1.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_1.development).__module__}.{type(var_1.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_1.file).__module__}.{type(var_1.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_1.hardware).__module__}.{type(var_1.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_1.internet).__module__}.{type(var_1.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_1.numeric).__module__}.{type(var_1.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_1.path).__module__}.{type(var_1.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_1.payment).__module__}.{type(var_1.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_1.science).__module__}.{type(var_1.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_1.transport).__module__}.{type(var_1.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = module_0.Generic()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_2.locale == module_1.Locale.EN
    assert f'{type(var_2.binaryfile).__module__}.{type(var_2.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_2.code).__module__}.{type(var_2.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_2.cryptographic).__module__}.{type(var_2.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_2.development).__module__}.{type(var_2.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_2.file).__module__}.{type(var_2.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_2.hardware).__module__}.{type(var_2.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_2.internet).__module__}.{type(var_2.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_2.numeric).__module__}.{type(var_2.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_2.path).__module__}.{type(var_2.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_2.payment).__module__}.{type(var_2.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_2.science).__module__}.{type(var_2.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_2.transport).__module__}.{type(var_2.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    var_2.__iadd__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.Generic()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == module_1.Locale.EN
    assert f'{type(var_0.binaryfile).__module__}.{type(var_0.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_0.code).__module__}.{type(var_0.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_0.cryptographic).__module__}.{type(var_0.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_0.development).__module__}.{type(var_0.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_0.file).__module__}.{type(var_0.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_0.hardware).__module__}.{type(var_0.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_0.internet).__module__}.{type(var_0.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_0.numeric).__module__}.{type(var_0.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.path).__module__}.{type(var_0.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_0.payment).__module__}.{type(var_0.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.science).__module__}.{type(var_0.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_0.transport).__module__}.{type(var_0.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.__str__()
    assert var_1 == 'Generic <Locale.EN>'
    var_2 = var_0.reseed(var_1)
    assert var_0.seed == 'Generic <Locale.EN>'
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'
    var_3 = module_2.BaseProvider
    var_4 = var_0.__iadd__(var_3)
    assert f'{type(var_0.baseprovider).__module__}.{type(var_0.baseprovider).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert var_4.seed == 'Generic <Locale.EN>'
    assert var_4.locale == module_1.Locale.EN
    assert f'{type(var_4.binaryfile).__module__}.{type(var_4.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_4.code).__module__}.{type(var_4.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_4.cryptographic).__module__}.{type(var_4.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_4.development).__module__}.{type(var_4.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_4.file).__module__}.{type(var_4.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_4.hardware).__module__}.{type(var_4.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_4.internet).__module__}.{type(var_4.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_4.numeric).__module__}.{type(var_4.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_4.path).__module__}.{type(var_4.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_4.payment).__module__}.{type(var_4.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_4.science).__module__}.{type(var_4.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_4.transport).__module__}.{type(var_4.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(var_4.address).__module__}.{type(var_4.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_4.datetime).__module__}.{type(var_4.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_4.finance).__module__}.{type(var_4.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_4.food).__module__}.{type(var_4.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_4.text).__module__}.{type(var_4.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_4.person).__module__}.{type(var_4.person).__qualname__}' == 'mimesis.providers.person.Person'
    assert f'{type(var_4.baseprovider).__module__}.{type(var_4.baseprovider).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(module_2.DATADIR).__module__}.{type(module_2.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.LOCALE_SEP == '-'
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'
    var_5 = None
    var_6 = [var_5, var_5, var_5]
    var_4.add_providers(*var_6)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.Generic()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == module_1.Locale.EN
    assert f'{type(var_0.binaryfile).__module__}.{type(var_0.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_0.code).__module__}.{type(var_0.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_0.cryptographic).__module__}.{type(var_0.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_0.development).__module__}.{type(var_0.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_0.file).__module__}.{type(var_0.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_0.hardware).__module__}.{type(var_0.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_0.internet).__module__}.{type(var_0.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_0.numeric).__module__}.{type(var_0.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.path).__module__}.{type(var_0.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_0.payment).__module__}.{type(var_0.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.science).__module__}.{type(var_0.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_0.transport).__module__}.{type(var_0.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = module_3.MatchValue
    var_0.__iadd__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.Generic(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale is None
    assert f'{type(var_1.binaryfile).__module__}.{type(var_1.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_1.code).__module__}.{type(var_1.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_1.cryptographic).__module__}.{type(var_1.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_1.development).__module__}.{type(var_1.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_1.file).__module__}.{type(var_1.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_1.hardware).__module__}.{type(var_1.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_1.internet).__module__}.{type(var_1.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_1.numeric).__module__}.{type(var_1.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_1.path).__module__}.{type(var_1.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_1.payment).__module__}.{type(var_1.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_1.science).__module__}.{type(var_1.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_1.transport).__module__}.{type(var_1.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1.reseed(var_0)

def test_case_8():
    var_0 = module_0.Generic()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == module_1.Locale.EN
    assert f'{type(var_0.binaryfile).__module__}.{type(var_0.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_0.code).__module__}.{type(var_0.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_0.cryptographic).__module__}.{type(var_0.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_0.development).__module__}.{type(var_0.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_0.file).__module__}.{type(var_0.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_0.hardware).__module__}.{type(var_0.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_0.internet).__module__}.{type(var_0.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_0.numeric).__module__}.{type(var_0.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.path).__module__}.{type(var_0.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_0.payment).__module__}.{type(var_0.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.science).__module__}.{type(var_0.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_0.transport).__module__}.{type(var_0.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = module_2.BaseProvider
    var_2 = var_0.__iadd__(var_1)
    assert f'{type(var_0.baseprovider).__module__}.{type(var_0.baseprovider).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_2.locale == module_1.Locale.EN
    assert f'{type(var_2.binaryfile).__module__}.{type(var_2.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_2.code).__module__}.{type(var_2.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_2.cryptographic).__module__}.{type(var_2.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_2.development).__module__}.{type(var_2.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_2.file).__module__}.{type(var_2.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_2.hardware).__module__}.{type(var_2.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_2.internet).__module__}.{type(var_2.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_2.numeric).__module__}.{type(var_2.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_2.path).__module__}.{type(var_2.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_2.payment).__module__}.{type(var_2.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_2.science).__module__}.{type(var_2.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_2.transport).__module__}.{type(var_2.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(var_2.baseprovider).__module__}.{type(var_2.baseprovider).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(module_2.DATADIR).__module__}.{type(module_2.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.LOCALE_SEP == '-'
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.Generic()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == module_1.Locale.EN
    assert f'{type(var_0.binaryfile).__module__}.{type(var_0.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_0.code).__module__}.{type(var_0.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_0.cryptographic).__module__}.{type(var_0.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_0.development).__module__}.{type(var_0.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_0.file).__module__}.{type(var_0.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_0.hardware).__module__}.{type(var_0.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_0.internet).__module__}.{type(var_0.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_0.numeric).__module__}.{type(var_0.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.path).__module__}.{type(var_0.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_0.payment).__module__}.{type(var_0.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.science).__module__}.{type(var_0.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_0.transport).__module__}.{type(var_0.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = module_0.Generic
    var_0.__iadd__(var_1)