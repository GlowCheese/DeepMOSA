# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.generic as module_0
import mimesis.enums as module_1
import ast as module_2
import inspect as module_3
import mimesis.providers.base as module_4

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
    var_1 = module_2.Assert
    with pytest.raises(TypeError):
        var_0.add_provider(var_1)

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
    var_3 = module_2.MatchMapping
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
    var_2 = module_0.Generic
    with pytest.raises(TypeError):
        var_0.add_provider(var_2)

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
    var_1 = var_0.reseed(var_0)
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'

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
    var_2 = var_0.add_providers()
    var_3 = var_0.reseed()
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_4 = None
    with pytest.raises(AttributeError):
        module_3.getattr_static(var_1, var_4)

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
    var_1 = '_non_callable'
    var_2 = 'not a function'
    var_3 = setattr(var_0, var_1, var_2)
    var_4 = 'non_callable'
    var_5 = var_0.__getattr__(var_4)
    assert var_5 is None

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'K.-Ff'
    var_1 = module_0.Generic(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'K.-Ff'
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
    var_2 = 22222
    var_1.reseed(var_2)

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
    var_1 = '_test_provider'
    var_2 = None
    var_3 = setattr(var_0, var_1, var_2)
    var_4 = 11111
    var_5 = var_0.reseed(var_4)
    assert var_0.seed == 11111
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'
    var_6 = var_0.seed
    assert var_6 == 11111

@pytest.mark.xfail(strict=True)
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
    var_1 = var_0.reseed()
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'
    var_2 = module_4.BaseProvider
    var_3 = var_0.__iadd__(var_2)
    assert f'{type(var_0.baseprovider).__module__}.{type(var_0.baseprovider).__qualname__}' == 'mimesis.providers.base.BaseProvider'
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
    assert f'{type(var_3.address).__module__}.{type(var_3.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_3.datetime).__module__}.{type(var_3.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_3.finance).__module__}.{type(var_3.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_3.food).__module__}.{type(var_3.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_3.text).__module__}.{type(var_3.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_3.person).__module__}.{type(var_3.person).__qualname__}' == 'mimesis.providers.person.Person'
    assert f'{type(var_3.baseprovider).__module__}.{type(var_3.baseprovider).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(module_4.DATADIR).__module__}.{type(module_4.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.LOCALE_SEP == '-'
    assert f'{type(module_4.MissingSeed).__module__}.{type(module_4.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_4.Seed).__module__}.{type(module_4.Seed).__qualname__}' == 'types.UnionType'
    var_4 = None
    module_3.getsourcefile(var_4)