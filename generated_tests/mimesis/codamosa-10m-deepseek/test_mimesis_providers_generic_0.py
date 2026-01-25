# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.generic as module_0
import mimesis.enums as module_1
import mimesis.providers.base as module_2
import re as module_3
import ast as module_4

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
    var_2 = var_0.reseed()
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'
    var_3 = module_0.Generic(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_3.locale is None
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
    var_2 = var_1.__str__()
    assert var_2 == 'Generic <None>'
    var_3 = module_0.Generic(seed=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert var_3.seed == 'Generic <None>'
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
    var_1.__iadd__(var_2)

def test_case_5():
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
    var_2 = module_2.BaseProvider
    var_3 = var_1.__iadd__(var_2)
    assert f'{type(var_1.baseprovider).__module__}.{type(var_1.baseprovider).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_3.locale is None
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
    assert f'{type(var_3.baseprovider).__module__}.{type(var_3.baseprovider).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(module_2.DATADIR).__module__}.{type(module_2.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.LOCALE_SEP == '-'
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'

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
    var_1 = module_0.Generic()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == module_1.Locale.EN
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
    var_2 = var_0.__str__()
    assert var_2 == 'Generic <Locale.EN>'
    var_3 = module_3.RegexFlag
    var_0.__iadd__(var_3)

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
    var_1 = var_0.reseed()
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'
    var_2 = module_4.AugStore
    var_3 = [var_2, var_2, var_2, var_2]
    var_0.add_providers(*var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
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
    var_1.reseed()

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
    var_1 = var_0.add_providers()
    var_2 = var_0.__dir__()
    var_3 = None
    var_4 = module_0.Generic(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'mimesis.types._MissingSeed'
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
    var_5 = module_0.Generic()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_5.seed).__module__}.{type(var_5.seed).__qualname__}' == 'mimesis.types._MissingSeed'
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
    var_6 = module_0.Generic
    var_5.__iadd__(var_6)