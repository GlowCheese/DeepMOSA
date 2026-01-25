####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_certificate_fingerprint_default_algorithm():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    assert isinstance(result, str)
    assert len(result) == 95
    assert result.count(":") == 31
    assert all(c in "0123456789ABCDEF:" for c in result)


def test_certificate_fingerprint_sha256():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint(algorithm="sha256")
    assert isinstance(result, str)
    assert len(result) == 95
    assert result.count(":") == 31
    assert all(c in "0123456789ABCDEF:" for c in result)


def test_certificate_fingerprint_sha1():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(result, str)
    assert len(result) == 59
    assert result.count(":") == 19
    assert all(c in "0123456789ABCDEF:" for c in result)


def test_certificate_fingerprint_invalid_algorithm():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    try:
        crypto.certificate_fingerprint(algorithm="md5")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


def test_certificate_fingerprint_format():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    parts = result.split(":")
    assert len(parts) == 32
    assert all(len(part) == 2 for part in parts)


def test_certificate_fingerprint_uppercase():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    assert result == result.upper()


def test_certificate_fingerprint_randomness():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result1 = crypto.certificate_fingerprint()
    result2 = crypto.certificate_fingerprint()
    assert result1 != result2


# LLM-generated content at query #2
#--------------------------

```python
def test_api_key_default():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 64
    assert all(c in '0123456789abcdef' for c in key)

def test_api_key_with_prefix():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    key = crypto.api_key(prefix='sk_')
    assert isinstance(key, str)
    assert key.startswith('sk_')
    assert len(key) == 67

def test_api_key_with_custom_length():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    key = crypto.api_key(length=16)
    assert isinstance(key, str)
    assert len(key) == 32

def test_api_key_with_prefix_and_length():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    key = crypto.api_key(prefix='api_', length=20)
    assert isinstance(key, str)
    assert key.startswith('api_')
    assert len(key) == 44

def test_api_key_hex_format():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    key = crypto.api_key(fmt='hex')
    assert isinstance(key, str)
    assert all(c in '0123456789abcdef' for c in key)

def test_api_key_base64_format():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    key = crypto.api_key(fmt='base64', length=32)
    assert isinstance(key, str)
    assert len(key) == 32

def test_api_key_base64_with_prefix():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    key = crypto.api_key(prefix='pk_', fmt='base64', length=24)
    assert isinstance(key, str)
    assert key.startswith('pk_')

def test_api_key_invalid_format():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    try:
        crypto.api_key(fmt='invalid')
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown format" in str(e)

def test_api_key_empty_prefix():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    key = crypto.api_key(prefix='')
    assert isinstance(key, str)
    assert len(key) == 64

def test_api_key_uniqueness():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    key1 = crypto.api_key()
    key2 = crypto.api_key()
    assert key1 != key2


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_api_key_default():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.api_key()
    assert isinstance(result, str)
    assert len(result) == 64


def test_api_key_with_prefix():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.api_key(prefix='sk_')
    assert isinstance(result, str)
    assert result.startswith('sk_')
    assert len(result) == 67


def test_api_key_custom_length():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.api_key(length=16)
    assert isinstance(result, str)
    assert len(result) == 32


def test_api_key_with_prefix_and_length():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.api_key(prefix='pk_', length=24)
    assert isinstance(result, str)
    assert result.startswith('pk_')
    assert len(result) == 51


def test_api_key_base64_format():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.api_key(fmt='base64')
    assert isinstance(result, str)
    assert len(result) <= 32


def test_api_key_base64_with_prefix():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.api_key(prefix='api_', fmt='base64')
    assert isinstance(result, str)
    assert result.startswith('api_')


def test_api_key_hex_format():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.api_key(fmt='hex')
    assert isinstance(result, str)
    assert len(result) == 64


def test_api_key_invalid_format():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    try:
        crypto.api_key(fmt='invalid')
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown format" in str(e)


def test_api_key_empty_prefix():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.api_key(prefix='')
    assert isinstance(result, str)
    assert len(result) == 64


def test_api_key_all_parameters():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    result = crypto.api_key(prefix='test_', length=20, fmt='hex')
    assert isinstance(result, str)
    assert result.startswith('test_')


# LLM-generated content at query #2
#--------------------------

```python
def test_certificate_fingerprint_sha256_default():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 95
    assert fingerprint.count(":") == 31
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)


def test_certificate_fingerprint_sha256_explicit():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint(algorithm="sha256")
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 95
    assert fingerprint.count(":") == 31
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)


def test_certificate_fingerprint_sha1():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 59
    assert fingerprint.count(":") == 19
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)


def test_certificate_fingerprint_invalid_algorithm():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    try:
        crypto.certificate_fingerprint(algorithm="md5")
        assert False, "ValueError should be raised"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


def test_certificate_fingerprint_format():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint()
    parts = fingerprint.split(":")
    assert len(parts) == 32
    assert all(len(part) == 2 for part in parts)


def test_certificate_fingerprint_uppercase():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint()
    assert fingerprint == fingerprint.upper()


def test_certificate_fingerprint_sha1_format():
    from mimesis import Cryptographic
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint(algorithm="sha1")
    parts = fingerprint.split(":")
    assert len(parts) == 20
    assert all(len(part) == 2 for part in parts)


