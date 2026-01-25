# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.generic as module_0
import mimesis.enums as module_1
import mimesis.providers.base as module_2
import ast as module_3
import re as module_4
import tokenize as module_5

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
    var_1 = module_2.BaseDataProvider
    var_2 = var_0.__iadd__(var_1)
    assert f'{type(var_0.basedataprovider).__module__}.{type(var_0.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
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
    assert f'{type(var_2.basedataprovider).__module__}.{type(var_2.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(module_2.DATADIR).__module__}.{type(module_2.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.LOCALE_SEP == '-'
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'

@pytest.mark.xfail(strict=True)
def test_case_2():
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
    var_1 = var_0.__dir__()
    var_2 = var_0.reseed(var_0)
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'
    var_0.__iadd__(var_1)

@pytest.mark.xfail(strict=True)
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
    var_1 = var_0.__str__()
    assert var_1 == 'Generic <Locale.EN>'
    var_2 = module_1.FileType
    var_3 = module_3.MatchMapping
    var_4 = [var_2, var_3]
    var_0.add_providers(*var_4)

def test_case_4():
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
    var_2 = module_2.BaseDataProvider
    var_3 = var_0.__iadd__(var_2)
    assert f'{type(var_0.basedataprovider).__module__}.{type(var_0.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_3.locale == module_1.Locale.EN
    assert f'{type(var_3.binaryfile).__module__}.{type(var_3.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_3.code).__module__}.{type(var_3.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_3.cryptographic).__module__}.{type(var_3.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_3.development).__module__}.{type(var_3.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_3.file).__module__}.{type(var_3.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_3.hardware).__module__}.{type(var_3.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_3.internet).__module__}.{type(var_3.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_3.numeric).__module__}.{type(var_3.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_3.path).__module__}.{type(var_3.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_3.payment).__module__}.{type(var_3.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_3.science).__module__}.{type(var_3.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_3.transport).__module__}.{type(var_3.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(var_3.basedataprovider).__module__}.{type(var_3.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(module_2.DATADIR).__module__}.{type(module_2.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.LOCALE_SEP == '-'
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'

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
    var_1 = module_3.unaryop
    var_0.__iadd__(var_1)

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
    var_1 = var_0.reseed(var_0)
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'

@pytest.mark.xfail(strict=True)
def test_case_7():
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
    var_1 = '_test_attr'
    var_2 = var_0.add_providers()
    var_3 = setattr(var_0, var_1, var_1)
    var_4 = 'test_attr'
    var_5 = var_0.__getattr__(var_4)
    assert var_5 is None
    var_0.__iadd__(var_5)

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
    var_1 = '_test_attr'
    var_2 = setattr(var_0, var_1, var_1)
    var_3 = 'test_attr'
    var_4 = var_0.__getattr__(var_3)
    assert var_4 is None

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
    var_1 = module_4.purge()
    assert module_4.ASCII == module_4.RegexFlag.ASCII
    assert module_4.A == module_4.RegexFlag.ASCII
    assert module_4.IGNORECASE == module_4.RegexFlag.IGNORECASE
    assert module_4.I == module_4.RegexFlag.IGNORECASE
    assert module_4.LOCALE == module_4.RegexFlag.LOCALE
    assert module_4.L == module_4.RegexFlag.LOCALE
    assert module_4.UNICODE == module_4.RegexFlag.UNICODE
    assert module_4.U == module_4.RegexFlag.UNICODE
    assert module_4.MULTILINE == module_4.RegexFlag.MULTILINE
    assert module_4.M == module_4.RegexFlag.MULTILINE
    assert module_4.DOTALL == module_4.RegexFlag.DOTALL
    assert module_4.S == module_4.RegexFlag.DOTALL
    assert module_4.VERBOSE == module_4.RegexFlag.VERBOSE
    assert module_4.X == module_4.RegexFlag.VERBOSE
    assert module_4.TEMPLATE == module_4.RegexFlag.TEMPLATE
    assert module_4.T == module_4.RegexFlag.TEMPLATE
    assert module_4.DEBUG == module_4.RegexFlag.DEBUG
    var_2 = var_0.reseed()
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'
    var_3 = module_5.generate_tokens(var_2)
    assert module_5.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_5.tok_name == {0: 'ENDMARKER', 1: 'NAME', 2: 'NUMBER', 3: 'STRING', 4: 'NEWLINE', 5: 'INDENT', 6: 'DEDENT', 7: 'LPAR', 8: 'RPAR', 9: 'LSQB', 10: 'RSQB', 11: 'COLON', 12: 'COMMA', 13: 'SEMI', 14: 'PLUS', 15: 'MINUS', 16: 'STAR', 17: 'SLASH', 18: 'VBAR', 19: 'AMPER', 20: 'LESS', 21: 'GREATER', 22: 'EQUAL', 23: 'DOT', 24: 'PERCENT', 25: 'LBRACE', 26: 'RBRACE', 27: 'EQEQUAL', 28: 'NOTEQUAL', 29: 'LESSEQUAL', 30: 'GREATEREQUAL', 31: 'TILDE', 32: 'CIRCUMFLEX', 33: 'LEFTSHIFT', 34: 'RIGHTSHIFT', 35: 'DOUBLESTAR', 36: 'PLUSEQUAL', 37: 'MINEQUAL', 38: 'STAREQUAL', 39: 'SLASHEQUAL', 40: 'PERCENTEQUAL', 41: 'AMPEREQUAL', 42: 'VBAREQUAL', 43: 'CIRCUMFLEXEQUAL', 44: 'LEFTSHIFTEQUAL', 45: 'RIGHTSHIFTEQUAL', 46: 'DOUBLESTAREQUAL', 47: 'DOUBLESLASH', 48: 'DOUBLESLASHEQUAL', 49: 'AT', 50: 'ATEQUAL', 51: 'RARROW', 52: 'ELLIPSIS', 53: 'COLONEQUAL', 54: 'OP', 55: 'AWAIT', 56: 'ASYNC', 57: 'TYPE_IGNORE', 58: 'TYPE_COMMENT', 59: 'SOFT_KEYWORD', 60: 'ERRORTOKEN', 61: 'COMMENT', 62: 'NL', 63: 'ENCODING', 64: 'N_TOKENS', 256: 'NT_OFFSET'}
    assert module_5.ENDMARKER == 0
    assert module_5.NAME == 1
    assert module_5.NUMBER == 2
    assert module_5.STRING == 3
    assert module_5.NEWLINE == 4
    assert module_5.INDENT == 5
    assert module_5.DEDENT == 6
    assert module_5.LPAR == 7
    assert module_5.RPAR == 8
    assert module_5.LSQB == 9
    assert module_5.RSQB == 10
    assert module_5.COLON == 11
    assert module_5.COMMA == 12
    assert module_5.SEMI == 13
    assert module_5.PLUS == 14
    assert module_5.MINUS == 15
    assert module_5.STAR == 16
    assert module_5.SLASH == 17
    assert module_5.VBAR == 18
    assert module_5.AMPER == 19
    assert module_5.LESS == 20
    assert module_5.GREATER == 21
    assert module_5.EQUAL == 22
    assert module_5.DOT == 23
    assert module_5.PERCENT == 24
    assert module_5.LBRACE == 25
    assert module_5.RBRACE == 26
    assert module_5.EQEQUAL == 27
    assert module_5.NOTEQUAL == 28
    assert module_5.LESSEQUAL == 29
    assert module_5.GREATEREQUAL == 30
    assert module_5.TILDE == 31
    assert module_5.CIRCUMFLEX == 32
    assert module_5.LEFTSHIFT == 33
    assert module_5.RIGHTSHIFT == 34
    assert module_5.DOUBLESTAR == 35
    assert module_5.PLUSEQUAL == 36
    assert module_5.MINEQUAL == 37
    assert module_5.STAREQUAL == 38
    assert module_5.SLASHEQUAL == 39
    assert module_5.PERCENTEQUAL == 40
    assert module_5.AMPEREQUAL == 41
    assert module_5.VBAREQUAL == 42
    assert module_5.CIRCUMFLEXEQUAL == 43
    assert module_5.LEFTSHIFTEQUAL == 44
    assert module_5.RIGHTSHIFTEQUAL == 45
    assert module_5.DOUBLESTAREQUAL == 46
    assert module_5.DOUBLESLASH == 47
    assert module_5.DOUBLESLASHEQUAL == 48
    assert module_5.AT == 49
    assert module_5.ATEQUAL == 50
    assert module_5.RARROW == 51
    assert module_5.ELLIPSIS == 52
    assert module_5.COLONEQUAL == 53
    assert module_5.OP == 54
    assert module_5.AWAIT == 55
    assert module_5.ASYNC == 56
    assert module_5.TYPE_IGNORE == 57
    assert module_5.TYPE_COMMENT == 58
    assert module_5.SOFT_KEYWORD == 59
    assert module_5.ERRORTOKEN == 60
    assert module_5.COMMENT == 61
    assert module_5.NL == 62
    assert module_5.ENCODING == 63
    assert module_5.N_TOKENS == 64
    assert module_5.NT_OFFSET == 256
    assert module_5.EXACT_TOKEN_TYPES == {'!=': 28, '%': 24, '%=': 40, '&': 19, '&=': 41, '(': 7, ')': 8, '*': 16, '**': 35, '**=': 46, '*=': 38, '+': 14, '+=': 36, ',': 12, '-': 15, '-=': 37, '->': 51, '.': 23, '...': 52, '/': 17, '//': 47, '//=': 48, '/=': 39, ':': 11, ':=': 53, ';': 13, '<': 20, '<<': 33, '<<=': 44, '<=': 29, '=': 22, '==': 27, '>': 21, '>=': 30, '>>': 34, '>>=': 45, '@': 49, '@=': 50, '[': 9, ']': 10, '^': 32, '^=': 43, '{': 25, '|': 18, '|=': 42, '}': 26, '~': 31}
    assert f'{type(module_5.cookie_re).__module__}.{type(module_5.cookie_re).__qualname__}' == 're.Pattern'
    assert f'{type(module_5.blank_re).__module__}.{type(module_5.blank_re).__qualname__}' == 're.Pattern'
    assert module_5.Whitespace == '[ \\f\\t]*'
    assert module_5.Comment == '#[^\\r\\n]*'
    assert module_5.Ignore == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?'
    assert module_5.Name == '\\w+'
    assert module_5.Hexnumber == '0[xX](?:_?[0-9a-fA-F])+'
    assert module_5.Binnumber == '0[bB](?:_?[01])+'
    assert module_5.Octnumber == '0[oO](?:_?[0-7])+'
    assert module_5.Decnumber == '(?:0(?:_?0)*|[1-9](?:_?[0-9])*)'
    assert module_5.Intnumber == '(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*))'
    assert module_5.Exponent == '[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_5.Pointfloat == '([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?'
    assert module_5.Expfloat == '[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_5.Floatnumber == '(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)'
    assert module_5.Imagnumber == '([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])'
    assert module_5.Number == '(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))'
    assert module_5.StringPrefix == '(|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)'
    assert module_5.Single == "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"
    assert module_5.Double == '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"'
    assert module_5.Single3 == "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''"
    assert module_5.Double3 == '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'
    assert module_5.Triple == '((|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)\'\'\'|(|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)""")'
    assert module_5.String == '((|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")'
    assert module_5.Special == '(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=)'
    assert module_5.Funny == '(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))'
    assert module_5.PlainToken == '((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_5.Token == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_5.ContStr == '((|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))'
    assert module_5.PseudoExtras == '(\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)\'\'\'|(|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)"""))'
    assert module_5.PseudoToken == '[ \\f\\t]*((\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)\'\'\'|(|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)"""))|(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|RB|fr|Rf|br|B|Fr|BR|rf|fR|U|rF|bR|u|Br|FR|b|rB|F|f|r|rb|RF|R|Rb)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))|\\w+)'
    assert module_5.endpats == {"'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", '"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", '"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "B'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'B"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "B'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'B"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "BR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'BR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "BR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'BR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "U'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'U"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "U'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'U"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "bR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'bR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "bR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'bR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "u'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'u"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "u'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'u"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "FR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'FR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "FR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'FR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "b'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'b"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "b'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'b"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "F'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'F"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "F'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'F"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "f'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'f"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "f'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'f"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "r'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'r"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "r'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'r"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "R'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'R"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "R'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'R"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'}
    assert module_5.single_quoted == {'fR"', '"', 'fr"', 'Fr"', 'U"', "R'", "b'", 'rB"', "BR'", "Rb'", "F'", 'BR"', 'b"', "bR'", 'B"', "Fr'", "B'", "rF'", "f'", 'RF"', 'br"', 'R"', 'Br"', "fr'", "FR'", "fR'", 'rb"', "Rf'", "u'", 'rF"', 'f"', "RB'", 'Rf"', 'rf"', 'F"', "rB'", 'u"', 'bR"', 'Rb"', 'RB"', "rb'", 'FR"', "RF'", "'", "br'", "Br'", "rf'", "U'", "r'", 'r"'}
    assert module_5.triple_quoted == {'U"""', "Fr'''", 'br"""', "br'''", 'fr"""', "BR'''", "U'''", "rB'''", "B'''", "RB'''", "rb'''", 'Br"""', 'FR"""', '"""', "u'''", 'R"""', 'Rf"""', 'rF"""', 'rB"""', "bR'''", "Br'''", "f'''", "R'''", "'''", 'rf"""', "fR'''", 'f"""', 'u"""', 'b"""', 'BR"""', "b'''", 'rb"""', 'Fr"""', "fr'''", 'F"""', 'r"""', 'fR"""', 'RF"""', 'RB"""', "rf'''", "rF'''", "Rf'''", "Rb'''", 'B"""', "RF'''", "F'''", "FR'''", "r'''", 'bR"""', 'Rb"""'}
    assert module_5.t == 'Rb'
    assert module_5.u == "Rb'''"
    assert module_5.tabsize == 8
    var_4 = module_0.Generic(var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'builtins.generator'
    assert var_4.locale is None
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
    var_5 = None
    var_4.reseed(var_5)

def test_case_10():
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
    var_1 = '_test_attr'
    var_2 = setattr(var_0, var_1, var_1)
    var_3 = var_0.reseed(var_2)
    assert var_0.seed is None
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'
    var_4 = module_2.BaseDataProvider
    var_5 = var_0.__iadd__(var_4)
    assert f'{type(var_0.basedataprovider).__module__}.{type(var_0.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert var_5.seed is None
    assert var_5.locale == module_1.Locale.EN
    assert f'{type(var_5.binaryfile).__module__}.{type(var_5.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_5.code).__module__}.{type(var_5.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_5.cryptographic).__module__}.{type(var_5.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_5.development).__module__}.{type(var_5.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_5.file).__module__}.{type(var_5.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_5.hardware).__module__}.{type(var_5.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_5.internet).__module__}.{type(var_5.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_5.numeric).__module__}.{type(var_5.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_5.path).__module__}.{type(var_5.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_5.payment).__module__}.{type(var_5.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_5.science).__module__}.{type(var_5.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_5.transport).__module__}.{type(var_5.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(var_5.address).__module__}.{type(var_5.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_5.datetime).__module__}.{type(var_5.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_5.finance).__module__}.{type(var_5.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_5.food).__module__}.{type(var_5.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_5.text).__module__}.{type(var_5.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_5.person).__module__}.{type(var_5.person).__qualname__}' == 'mimesis.providers.person.Person'
    assert f'{type(var_5.basedataprovider).__module__}.{type(var_5.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(module_2.DATADIR).__module__}.{type(module_2.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.LOCALE_SEP == '-'
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'

@pytest.mark.xfail(strict=True)
def test_case_11():
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
    var_1 = '_test_attr'
    var_2 = setattr(var_0, var_1, var_1)
    var_3 = var_0.reseed(var_2)
    assert var_0.seed is None
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'
    var_4 = module_2.BaseDataProvider
    var_5 = var_0.__iadd__(var_4)
    assert f'{type(var_0.basedataprovider).__module__}.{type(var_0.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert var_5.seed is None
    assert var_5.locale == module_1.Locale.EN
    assert f'{type(var_5.binaryfile).__module__}.{type(var_5.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_5.code).__module__}.{type(var_5.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_5.cryptographic).__module__}.{type(var_5.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_5.development).__module__}.{type(var_5.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_5.file).__module__}.{type(var_5.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_5.hardware).__module__}.{type(var_5.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_5.internet).__module__}.{type(var_5.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_5.numeric).__module__}.{type(var_5.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_5.path).__module__}.{type(var_5.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_5.payment).__module__}.{type(var_5.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_5.science).__module__}.{type(var_5.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_5.transport).__module__}.{type(var_5.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(var_5.address).__module__}.{type(var_5.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_5.datetime).__module__}.{type(var_5.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_5.finance).__module__}.{type(var_5.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_5.food).__module__}.{type(var_5.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_5.text).__module__}.{type(var_5.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_5.person).__module__}.{type(var_5.person).__qualname__}' == 'mimesis.providers.person.Person'
    assert f'{type(var_5.basedataprovider).__module__}.{type(var_5.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(module_2.DATADIR).__module__}.{type(module_2.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.LOCALE_SEP == '-'
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'
    var_6 = var_0.__dir__()
    var_7 = module_0.Generic
    var_0.__iadd__(var_7)