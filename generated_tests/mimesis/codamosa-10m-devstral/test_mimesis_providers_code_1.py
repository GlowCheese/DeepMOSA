# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.providers.code as module_0
import mimesis.enums as module_1

def test_case_0():
    var_0 = module_0.Code()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.EAN_MASKS == {'ean-8': '########', 'ean-13': '#############'}
    assert module_0.IMEI_TACS == ['01124500', '01161200', '01194800', '01233600', '01300600', '01332700', '35875505', '35881505', '35925406', '35325807', '35391805', '35824005', '35896704', '35803106', '35838706', '35929005', '35089080', '35154900', '35920605', '35699601', '35316004', '35909205', '35328504', '35332705', '35316605', '35744105', '32930400', '35171005', '35511405']
    assert module_0.ISBN_GROUPS == {'ar-ae': '9948', 'ar-bh': '99901', 'ar-dz': '9961', 'ar-eg': '977', 'ar-iq': '9922', 'ar-jo': '9957', 'ar-kw': '9921', 'ar-lb': '9953', 'ar-ly': '9959', 'ar-ma': '9954', 'ar-om': '99969', 'ar-qa': '9927', 'ar-ps': '9950', 'ar-sa': '9960', 'ar-sy': '9927', 'ar-tn': '9938', 'ar-ye': '0000', 'az': '9952', 'cs': '80', 'da': '87', 'de': '3', 'de-at': '3', 'de-ch': '3', 'el': '618', 'en': '1', 'en-au': '1', 'en-ca': '1', 'en-gb': '1', 'es': '84', 'es-mx': '607', 'et': '9949', 'fa': '600', 'fi': '951', 'fr': '2', 'hr': '385', 'hu': '615', 'is': '9935', 'it': '88', 'ja': '4', 'kk': '601', 'ko': '89', 'nl': '90', 'nl-be': '90', 'no': '82', 'pl': '83', 'pt': '972', 'pt-br': '85', 'ru': '5', 'sk': '80', 'sv': '91', 'tr': '605', 'uk': '966', 'zh': '7', 'default': '#'}
    assert module_0.ISBN_MASKS == {'isbn-13': '###-{0}-#####-###-#', 'isbn-10': '{0}-#####-###-#'}
    assert module_0.LOCALE_CODES == ['af', 'ar-ae', 'ar-bh', 'ar-dz', 'ar-eg', 'ar-iq', 'ar-jo', 'ar-kw', 'ar-lb', 'ar-ly', 'ar-ma', 'ar-om', 'ar-qa', 'ar-ps', 'ar-sa', 'ar-sy', 'ar-tn', 'ar-ye', 'be', 'bg', 'ca', 'cs', 'da', 'de', 'de-at', 'de-ch', 'de-li', 'de-lu', 'el', 'en', 'en-au', 'en-bz', 'en-ca', 'en-gb', 'en-ie', 'en-jm', 'en-nz', 'en-tt', 'en-us', 'en-za', 'es', 'es-ar', 'es-bo', 'es-cl', 'es-co', 'es-cr', 'es-do', 'es-ec', 'es-gt', 'es-hn', 'es-mx', 'es-ni', 'es-pa', 'es-pe', 'es-pr', 'es-py', 'es-sv', 'es-uy', 'es-ve', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'fr-be', 'fr-ca', 'fr-ch', 'fr-lu', 'ga', 'gd', 'he', 'hi', 'hr', 'hu', 'id', 'is', 'it', 'it-ch', 'ja', 'ji', 'ko', 'ko', 'lt', 'lv', 'mk', 'ms', 'mt', 'nl', 'nl-be', 'no', 'no', 'pl', 'pt', 'pt-br', 'rm', 'ro', 'ro-mo', 'ru', 'ru-mo', 'sb', 'sk', 'sl', 'sq', 'sr', 'sr', 'sv', 'sv-fi', 'sx', 'sz', 'th', 'tn', 'tr', 'ts', 'uk', 'ur', 've', 'vi', 'xh', 'zh-cn', 'zh-hk', 'zh-sg', 'zh-tw', 'zu']
    var_1 = var_0.ean()

def test_case_1():
    var_0 = None
    var_1 = module_1.EANFormat.EAN8
    var_2 = None
    var_3 = module_0.Code()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.EAN_MASKS == {'ean-8': '########', 'ean-13': '#############'}
    assert module_0.IMEI_TACS == ['01124500', '01161200', '01194800', '01233600', '01300600', '01332700', '35875505', '35881505', '35925406', '35325807', '35391805', '35824005', '35896704', '35803106', '35838706', '35929005', '35089080', '35154900', '35920605', '35699601', '35316004', '35909205', '35328504', '35332705', '35316605', '35744105', '32930400', '35171005', '35511405']
    assert module_0.ISBN_GROUPS == {'ar-ae': '9948', 'ar-bh': '99901', 'ar-dz': '9961', 'ar-eg': '977', 'ar-iq': '9922', 'ar-jo': '9957', 'ar-kw': '9921', 'ar-lb': '9953', 'ar-ly': '9959', 'ar-ma': '9954', 'ar-om': '99969', 'ar-qa': '9927', 'ar-ps': '9950', 'ar-sa': '9960', 'ar-sy': '9927', 'ar-tn': '9938', 'ar-ye': '0000', 'az': '9952', 'cs': '80', 'da': '87', 'de': '3', 'de-at': '3', 'de-ch': '3', 'el': '618', 'en': '1', 'en-au': '1', 'en-ca': '1', 'en-gb': '1', 'es': '84', 'es-mx': '607', 'et': '9949', 'fa': '600', 'fi': '951', 'fr': '2', 'hr': '385', 'hu': '615', 'is': '9935', 'it': '88', 'ja': '4', 'kk': '601', 'ko': '89', 'nl': '90', 'nl-be': '90', 'no': '82', 'pl': '83', 'pt': '972', 'pt-br': '85', 'ru': '5', 'sk': '80', 'sv': '91', 'tr': '605', 'uk': '966', 'zh': '7', 'default': '#'}
    assert module_0.ISBN_MASKS == {'isbn-13': '###-{0}-#####-###-#', 'isbn-10': '{0}-#####-###-#'}
    assert module_0.LOCALE_CODES == ['af', 'ar-ae', 'ar-bh', 'ar-dz', 'ar-eg', 'ar-iq', 'ar-jo', 'ar-kw', 'ar-lb', 'ar-ly', 'ar-ma', 'ar-om', 'ar-qa', 'ar-ps', 'ar-sa', 'ar-sy', 'ar-tn', 'ar-ye', 'be', 'bg', 'ca', 'cs', 'da', 'de', 'de-at', 'de-ch', 'de-li', 'de-lu', 'el', 'en', 'en-au', 'en-bz', 'en-ca', 'en-gb', 'en-ie', 'en-jm', 'en-nz', 'en-tt', 'en-us', 'en-za', 'es', 'es-ar', 'es-bo', 'es-cl', 'es-co', 'es-cr', 'es-do', 'es-ec', 'es-gt', 'es-hn', 'es-mx', 'es-ni', 'es-pa', 'es-pe', 'es-pr', 'es-py', 'es-sv', 'es-uy', 'es-ve', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'fr-be', 'fr-ca', 'fr-ch', 'fr-lu', 'ga', 'gd', 'he', 'hi', 'hr', 'hu', 'id', 'is', 'it', 'it-ch', 'ja', 'ji', 'ko', 'ko', 'lt', 'lv', 'mk', 'ms', 'mt', 'nl', 'nl-be', 'no', 'no', 'pl', 'pt', 'pt-br', 'rm', 'ro', 'ro-mo', 'ru', 'ru-mo', 'sb', 'sk', 'sl', 'sq', 'sr', 'sr', 'sv', 'sv-fi', 'sx', 'sz', 'th', 'tn', 'tr', 'ts', 'uk', 'ur', 've', 'vi', 'xh', 'zh-cn', 'zh-hk', 'zh-sg', 'zh-tw', 'zu']
    var_4 = module_0.Code(seed=var_1, random=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert var_4.seed == module_1.EANFormat.EAN8
    var_5 = var_4.ean(var_0)
    assert var_5 == '2139118672408'
    var_6 = var_3.locale_code()

def test_case_2():
    var_0 = module_0.Code()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.EAN_MASKS == {'ean-8': '########', 'ean-13': '#############'}
    assert module_0.IMEI_TACS == ['01124500', '01161200', '01194800', '01233600', '01300600', '01332700', '35875505', '35881505', '35925406', '35325807', '35391805', '35824005', '35896704', '35803106', '35838706', '35929005', '35089080', '35154900', '35920605', '35699601', '35316004', '35909205', '35328504', '35332705', '35316605', '35744105', '32930400', '35171005', '35511405']
    assert module_0.ISBN_GROUPS == {'ar-ae': '9948', 'ar-bh': '99901', 'ar-dz': '9961', 'ar-eg': '977', 'ar-iq': '9922', 'ar-jo': '9957', 'ar-kw': '9921', 'ar-lb': '9953', 'ar-ly': '9959', 'ar-ma': '9954', 'ar-om': '99969', 'ar-qa': '9927', 'ar-ps': '9950', 'ar-sa': '9960', 'ar-sy': '9927', 'ar-tn': '9938', 'ar-ye': '0000', 'az': '9952', 'cs': '80', 'da': '87', 'de': '3', 'de-at': '3', 'de-ch': '3', 'el': '618', 'en': '1', 'en-au': '1', 'en-ca': '1', 'en-gb': '1', 'es': '84', 'es-mx': '607', 'et': '9949', 'fa': '600', 'fi': '951', 'fr': '2', 'hr': '385', 'hu': '615', 'is': '9935', 'it': '88', 'ja': '4', 'kk': '601', 'ko': '89', 'nl': '90', 'nl-be': '90', 'no': '82', 'pl': '83', 'pt': '972', 'pt-br': '85', 'ru': '5', 'sk': '80', 'sv': '91', 'tr': '605', 'uk': '966', 'zh': '7', 'default': '#'}
    assert module_0.ISBN_MASKS == {'isbn-13': '###-{0}-#####-###-#', 'isbn-10': '{0}-#####-###-#'}
    assert module_0.LOCALE_CODES == ['af', 'ar-ae', 'ar-bh', 'ar-dz', 'ar-eg', 'ar-iq', 'ar-jo', 'ar-kw', 'ar-lb', 'ar-ly', 'ar-ma', 'ar-om', 'ar-qa', 'ar-ps', 'ar-sa', 'ar-sy', 'ar-tn', 'ar-ye', 'be', 'bg', 'ca', 'cs', 'da', 'de', 'de-at', 'de-ch', 'de-li', 'de-lu', 'el', 'en', 'en-au', 'en-bz', 'en-ca', 'en-gb', 'en-ie', 'en-jm', 'en-nz', 'en-tt', 'en-us', 'en-za', 'es', 'es-ar', 'es-bo', 'es-cl', 'es-co', 'es-cr', 'es-do', 'es-ec', 'es-gt', 'es-hn', 'es-mx', 'es-ni', 'es-pa', 'es-pe', 'es-pr', 'es-py', 'es-sv', 'es-uy', 'es-ve', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'fr-be', 'fr-ca', 'fr-ch', 'fr-lu', 'ga', 'gd', 'he', 'hi', 'hr', 'hu', 'id', 'is', 'it', 'it-ch', 'ja', 'ji', 'ko', 'ko', 'lt', 'lv', 'mk', 'ms', 'mt', 'nl', 'nl-be', 'no', 'no', 'pl', 'pt', 'pt-br', 'rm', 'ro', 'ro-mo', 'ru', 'ru-mo', 'sb', 'sk', 'sl', 'sq', 'sr', 'sr', 'sv', 'sv-fi', 'sx', 'sz', 'th', 'tn', 'tr', 'ts', 'uk', 'ur', 've', 'vi', 'xh', 'zh-cn', 'zh-hk', 'zh-sg', 'zh-tw', 'zu']
    var_1 = var_0.issn()
    var_2 = var_0.pin()
    var_3 = var_0.ean()

def test_case_3():
    var_0 = None
    var_1 = module_0.Code()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.EAN_MASKS == {'ean-8': '########', 'ean-13': '#############'}
    assert module_0.IMEI_TACS == ['01124500', '01161200', '01194800', '01233600', '01300600', '01332700', '35875505', '35881505', '35925406', '35325807', '35391805', '35824005', '35896704', '35803106', '35838706', '35929005', '35089080', '35154900', '35920605', '35699601', '35316004', '35909205', '35328504', '35332705', '35316605', '35744105', '32930400', '35171005', '35511405']
    assert module_0.ISBN_GROUPS == {'ar-ae': '9948', 'ar-bh': '99901', 'ar-dz': '9961', 'ar-eg': '977', 'ar-iq': '9922', 'ar-jo': '9957', 'ar-kw': '9921', 'ar-lb': '9953', 'ar-ly': '9959', 'ar-ma': '9954', 'ar-om': '99969', 'ar-qa': '9927', 'ar-ps': '9950', 'ar-sa': '9960', 'ar-sy': '9927', 'ar-tn': '9938', 'ar-ye': '0000', 'az': '9952', 'cs': '80', 'da': '87', 'de': '3', 'de-at': '3', 'de-ch': '3', 'el': '618', 'en': '1', 'en-au': '1', 'en-ca': '1', 'en-gb': '1', 'es': '84', 'es-mx': '607', 'et': '9949', 'fa': '600', 'fi': '951', 'fr': '2', 'hr': '385', 'hu': '615', 'is': '9935', 'it': '88', 'ja': '4', 'kk': '601', 'ko': '89', 'nl': '90', 'nl-be': '90', 'no': '82', 'pl': '83', 'pt': '972', 'pt-br': '85', 'ru': '5', 'sk': '80', 'sv': '91', 'tr': '605', 'uk': '966', 'zh': '7', 'default': '#'}
    assert module_0.ISBN_MASKS == {'isbn-13': '###-{0}-#####-###-#', 'isbn-10': '{0}-#####-###-#'}
    assert module_0.LOCALE_CODES == ['af', 'ar-ae', 'ar-bh', 'ar-dz', 'ar-eg', 'ar-iq', 'ar-jo', 'ar-kw', 'ar-lb', 'ar-ly', 'ar-ma', 'ar-om', 'ar-qa', 'ar-ps', 'ar-sa', 'ar-sy', 'ar-tn', 'ar-ye', 'be', 'bg', 'ca', 'cs', 'da', 'de', 'de-at', 'de-ch', 'de-li', 'de-lu', 'el', 'en', 'en-au', 'en-bz', 'en-ca', 'en-gb', 'en-ie', 'en-jm', 'en-nz', 'en-tt', 'en-us', 'en-za', 'es', 'es-ar', 'es-bo', 'es-cl', 'es-co', 'es-cr', 'es-do', 'es-ec', 'es-gt', 'es-hn', 'es-mx', 'es-ni', 'es-pa', 'es-pe', 'es-pr', 'es-py', 'es-sv', 'es-uy', 'es-ve', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'fr-be', 'fr-ca', 'fr-ch', 'fr-lu', 'ga', 'gd', 'he', 'hi', 'hr', 'hu', 'id', 'is', 'it', 'it-ch', 'ja', 'ji', 'ko', 'ko', 'lt', 'lv', 'mk', 'ms', 'mt', 'nl', 'nl-be', 'no', 'no', 'pl', 'pt', 'pt-br', 'rm', 'ro', 'ro-mo', 'ru', 'ru-mo', 'sb', 'sk', 'sl', 'sq', 'sr', 'sr', 'sv', 'sv-fi', 'sx', 'sz', 'th', 'tn', 'tr', 'ts', 'uk', 'ur', 've', 'vi', 'xh', 'zh-cn', 'zh-hk', 'zh-sg', 'zh-tw', 'zu']
    var_2 = var_1.isbn(var_0)
    var_3 = 'c*p#'
    var_4 = module_0.Code()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_5 = module_1.Locale.SK
    var_6 = var_4.isbn(locale=var_5)
    var_7 = var_4.issn(var_3)

def test_case_4():
    var_0 = '\x0cHz`a0'
    var_1 = 'Ki\tnA6*7gcsNMV~'
    var_2 = module_0.Code()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.EAN_MASKS == {'ean-8': '########', 'ean-13': '#############'}
    assert module_0.IMEI_TACS == ['01124500', '01161200', '01194800', '01233600', '01300600', '01332700', '35875505', '35881505', '35925406', '35325807', '35391805', '35824005', '35896704', '35803106', '35838706', '35929005', '35089080', '35154900', '35920605', '35699601', '35316004', '35909205', '35328504', '35332705', '35316605', '35744105', '32930400', '35171005', '35511405']
    assert module_0.ISBN_GROUPS == {'ar-ae': '9948', 'ar-bh': '99901', 'ar-dz': '9961', 'ar-eg': '977', 'ar-iq': '9922', 'ar-jo': '9957', 'ar-kw': '9921', 'ar-lb': '9953', 'ar-ly': '9959', 'ar-ma': '9954', 'ar-om': '99969', 'ar-qa': '9927', 'ar-ps': '9950', 'ar-sa': '9960', 'ar-sy': '9927', 'ar-tn': '9938', 'ar-ye': '0000', 'az': '9952', 'cs': '80', 'da': '87', 'de': '3', 'de-at': '3', 'de-ch': '3', 'el': '618', 'en': '1', 'en-au': '1', 'en-ca': '1', 'en-gb': '1', 'es': '84', 'es-mx': '607', 'et': '9949', 'fa': '600', 'fi': '951', 'fr': '2', 'hr': '385', 'hu': '615', 'is': '9935', 'it': '88', 'ja': '4', 'kk': '601', 'ko': '89', 'nl': '90', 'nl-be': '90', 'no': '82', 'pl': '83', 'pt': '972', 'pt-br': '85', 'ru': '5', 'sk': '80', 'sv': '91', 'tr': '605', 'uk': '966', 'zh': '7', 'default': '#'}
    assert module_0.ISBN_MASKS == {'isbn-13': '###-{0}-#####-###-#', 'isbn-10': '{0}-#####-###-#'}
    assert module_0.LOCALE_CODES == ['af', 'ar-ae', 'ar-bh', 'ar-dz', 'ar-eg', 'ar-iq', 'ar-jo', 'ar-kw', 'ar-lb', 'ar-ly', 'ar-ma', 'ar-om', 'ar-qa', 'ar-ps', 'ar-sa', 'ar-sy', 'ar-tn', 'ar-ye', 'be', 'bg', 'ca', 'cs', 'da', 'de', 'de-at', 'de-ch', 'de-li', 'de-lu', 'el', 'en', 'en-au', 'en-bz', 'en-ca', 'en-gb', 'en-ie', 'en-jm', 'en-nz', 'en-tt', 'en-us', 'en-za', 'es', 'es-ar', 'es-bo', 'es-cl', 'es-co', 'es-cr', 'es-do', 'es-ec', 'es-gt', 'es-hn', 'es-mx', 'es-ni', 'es-pa', 'es-pe', 'es-pr', 'es-py', 'es-sv', 'es-uy', 'es-ve', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'fr-be', 'fr-ca', 'fr-ch', 'fr-lu', 'ga', 'gd', 'he', 'hi', 'hr', 'hu', 'id', 'is', 'it', 'it-ch', 'ja', 'ji', 'ko', 'ko', 'lt', 'lv', 'mk', 'ms', 'mt', 'nl', 'nl-be', 'no', 'no', 'pl', 'pt', 'pt-br', 'rm', 'ro', 'ro-mo', 'ru', 'ru-mo', 'sb', 'sk', 'sl', 'sq', 'sr', 'sr', 'sv', 'sv-fi', 'sx', 'sz', 'th', 'tn', 'tr', 'ts', 'uk', 'ur', 've', 'vi', 'xh', 'zh-cn', 'zh-hk', 'zh-sg', 'zh-tw', 'zu']
    var_3 = var_2.issn()
    var_4 = module_0.Code(seed=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert var_4.seed == 'Ki\tnA6*7gcsNMV~'
    var_5 = var_4.pin(var_0)
    assert var_5 == '\x0cHz`a0'
    var_6 = None
    var_7 = module_0.Code(seed=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'mimesis.providers.code.Code'
    assert f'{type(var_7.random).__module__}.{type(var_7.random).__qualname__}' == 'mimesis.random.Random'
    assert var_7.seed is None
    var_8 = var_7.imei()