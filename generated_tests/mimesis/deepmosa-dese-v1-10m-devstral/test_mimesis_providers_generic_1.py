# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.generic as module_0
import mimesis.enums as module_1
import ast as module_2
import mimesis.providers.base as module_3
import inspect as module_4

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
    var_1 = module_2.Store
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
    var_1 = var_0.reseed()
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'
    var_2 = module_1.FileType
    var_3 = module_3.BaseDataProvider
    var_4 = var_0.__iadd__(var_3)
    assert f'{type(var_0.basedataprovider).__module__}.{type(var_0.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'mimesis.types._MissingSeed'
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
    assert f'{type(var_4.basedataprovider).__module__}.{type(var_4.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(module_3.DATADIR).__module__}.{type(module_3.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_3.LOCALE_SEP == '-'
    assert f'{type(module_3.MissingSeed).__module__}.{type(module_3.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_3.Seed).__module__}.{type(module_3.Seed).__qualname__}' == 'types.UnionType'
    var_5 = [var_2, var_3]
    var_0.add_providers(*var_5)

@pytest.mark.xfail(strict=True)
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
    var_2 = module_4.Attribute
    var_0.__iadd__(var_2)

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
        module_4.getattr_static(var_1, var_4)

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
    var_1 = module_3.BaseDataProvider
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
    assert f'{type(module_3.DATADIR).__module__}.{type(module_3.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_3.LOCALE_SEP == '-'
    assert f'{type(module_3.MissingSeed).__module__}.{type(module_3.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_3.Seed).__module__}.{type(module_3.Seed).__qualname__}' == 'types.UnionType'

@pytest.mark.xfail(strict=True)
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
    var_1 = var_0.__str__()
    assert var_1 == 'Generic <Locale.EN>'
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    var_4 = var_0.reseed()
    assert f'{type(var_0.address).__module__}.{type(var_0.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_0.datetime).__module__}.{type(var_0.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_0.finance).__module__}.{type(var_0.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_0.food).__module__}.{type(var_0.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.text).__module__}.{type(var_0.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.person).__module__}.{type(var_0.person).__qualname__}' == 'mimesis.providers.person.Person'
    var_5 = module_3.BaseDataProvider
    var_6 = var_0.__iadd__(var_5)
    assert f'{type(var_0.basedataprovider).__module__}.{type(var_0.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_6.random).__module__}.{type(var_6.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_6.seed).__module__}.{type(var_6.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_6.locale == module_1.Locale.EN
    assert f'{type(var_6.binaryfile).__module__}.{type(var_6.binaryfile).__qualname__}' == 'mimesis.providers.binaryfile.BinaryFile'
    assert f'{type(var_6.code).__module__}.{type(var_6.code).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_6.cryptographic).__module__}.{type(var_6.cryptographic).__qualname__}' == 'mimesis.providers.cryptographic.Cryptographic'
    assert f'{type(var_6.development).__module__}.{type(var_6.development).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_6.file).__module__}.{type(var_6.file).__qualname__}' == 'mimesis.providers.file.File'
    assert f'{type(var_6.hardware).__module__}.{type(var_6.hardware).__qualname__}' == 'mimesis.providers.hardware.Hardware'
    assert f'{type(var_6.internet).__module__}.{type(var_6.internet).__qualname__}' == 'mimesis.providers.internet.Internet'
    assert f'{type(var_6.numeric).__module__}.{type(var_6.numeric).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_6.path).__module__}.{type(var_6.path).__qualname__}' == 'mimesis.providers.path.Path'
    assert f'{type(var_6.payment).__module__}.{type(var_6.payment).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_6.science).__module__}.{type(var_6.science).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_6.transport).__module__}.{type(var_6.transport).__qualname__}' == 'mimesis.providers.transport.Transport'
    assert f'{type(var_6.address).__module__}.{type(var_6.address).__qualname__}' == 'mimesis.providers.address.Address'
    assert f'{type(var_6.datetime).__module__}.{type(var_6.datetime).__qualname__}' == 'mimesis.providers.date.Datetime'
    assert f'{type(var_6.finance).__module__}.{type(var_6.finance).__qualname__}' == 'mimesis.providers.finance.Finance'
    assert f'{type(var_6.food).__module__}.{type(var_6.food).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_6.text).__module__}.{type(var_6.text).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_6.person).__module__}.{type(var_6.person).__qualname__}' == 'mimesis.providers.person.Person'
    assert f'{type(var_6.basedataprovider).__module__}.{type(var_6.basedataprovider).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(module_3.DATADIR).__module__}.{type(module_3.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_3.LOCALE_SEP == '-'
    assert f'{type(module_3.MissingSeed).__module__}.{type(module_3.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_3.Seed).__module__}.{type(module_3.Seed).__qualname__}' == 'types.UnionType'
    var_7 = module_0.Generic
    var_6.__iadd__(var_7)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = module_0.Generic(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.generic.Generic'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed is None
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