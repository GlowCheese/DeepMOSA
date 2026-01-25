####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_certificate_fingerprint_default():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    assert isinstance(result, str)
    assert len(result) == 59  # 32 bytes * 2 chars + 31 colons
    assert all(c in "0123456789ABCDEF:" for c in result)
    assert result.count(":") == 31

def test_certificate_fingerprint_sha1():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(result, str)
    assert len(result) == 59  # 20 bytes * 2 chars + 19 colons
    assert all(c in "0123456789ABCDEF:" for c in result)
    assert result.count(":") == 19

def test_certificate_fingerprint_invalid_algorithm():
    crypto = Cryptographic()
    try:
        crypto.certificate_fingerprint(algorithm="md5")
    except ValueError as e:
        assert str(e) == "Unknown algorithm: md5. Use 'sha256' or 'sha1'."


# LLM-generated content at query #2
#--------------------------

```python
def test_api_key_default():
    crypto = Cryptographic()
    result = crypto.api_key()
    assert isinstance(result, str)
    assert len(result) == 32

def test_api_key_with_prefix():
    crypto = Cryptographic()
    result = crypto.api_key(prefix="sk_")
    assert result.startswith("sk_")
    assert len(result) == 35

def test_api_key_hex_format():
    crypto = Cryptographic()
    result = crypto.api_key(fmt="hex")
    assert all(c in "0123456789abcdef" for c in result)

def test_api_key_base64_format():
    crypto = Cryptographic()
    result = crypto.api_key(fmt="base64")
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=" for c in result)

def test_api_key_custom_length():
    crypto = Cryptographic()
    result = crypto.api_key(length=16)
    assert len(result) == 16

def test_api_key_invalid_format():
    crypto = Cryptographic()
    try:
        crypto.api_key(fmt="invalid")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Unknown format: invalid. Use 'hex' or 'base64'."


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_api_key_default():
    crypto = Cryptographic()
    result = crypto.api_key()
    assert isinstance(result, str)
    assert len(result) == 32
    assert all(c in "0123456789abcdef" for c in result)

def test_api_key_with_prefix():
    crypto = Cryptographic()
    result = crypto.api_key(prefix="sk_")
    assert isinstance(result, str)
    assert result.startswith("sk_")
    assert len(result) == 35
    assert all(c in "0123456789abcdef" for c in result[3:])

def test_api_key_hex_format():
    crypto = Cryptographic()
    result = crypto.api_key(fmt="hex")
    assert isinstance(result, str)
    assert len(result) == 32
    assert all(c in "0123456789abcdef" for c in result)

def test_api_key_base64_format():
    crypto = Cryptographic()
    result = crypto.api_key(fmt="base64")
    assert isinstance(result, str)
    assert len(result) == 32
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=" for c in result)

def test_api_key_custom_length():
    crypto = Cryptographic()
    result = crypto.api_key(length=16, fmt="hex")
    assert isinstance(result, str)
    assert len(result) == 16
    assert all(c in "0123456789abcdef" for c in result)

def test_api_key_invalid_format():
    crypto = Cryptographic()
    try:
        crypto.api_key(fmt="invalid")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Unknown format: invalid. Use 'hex' or 'base64'."


# LLM-generated content at query #2
#--------------------------

```python
def test_certificate_fingerprint_default():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    assert isinstance(result, str)
    assert len(result) == 2 * 32 + 31  # 32 bytes in SHA-256, colon-separated
    assert all(c in "0123456789ABCDEF:" for c in result)

def test_certificate_fingerprint_sha1():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(result, str)
    assert len(result) == 2 * 20 + 19  # 20 bytes in SHA-1, colon-separated
    assert all(c in "0123456789ABCDEF:" for c in result)

def test_certificate_fingerprint_invalid_algorithm():
    crypto = Cryptographic()
    try:
        crypto.certificate_fingerprint(algorithm="invalid")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Unknown algorithm: invalid. Use 'sha256' or 'sha1'."


