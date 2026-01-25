####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Cryptographic_api_key():
    crypto = Cryptographic()

    # Test default API key (hex format, no prefix)
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 32
    assert all(c in "0123456789abcdef" for c in key)

    # Test API key with prefix
    key_with_prefix = crypto.api_key(prefix="sk_")
    assert key_with_prefix.startswith("sk_")
    assert len(key_with_prefix) == 35  # 32 + len("sk_")
    assert all(c in "0123456789abcdef" for c in key_with_prefix[3:])

    # Test API key with base64 format
    key_base64 = crypto.api_key(fmt="base64")
    assert isinstance(key_base64, str)
    assert len(key_base64) == 32
    # Base64 can contain A-Z, a-z, 0-9, '-', '_'
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in key_base64)

    # Test API key with prefix and base64 format
    key_prefix_base64 = crypto.api_key(prefix="pk_", fmt="base64")
    assert key_prefix_base64.startswith("pk_")
    assert len(key_prefix_base64) == 35  # 32 + len("pk_")
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in key_prefix_base64[3:])

    # Test invalid format
    try:
        crypto.api_key(fmt="invalid")
        assert False, "Expected ValueError for invalid format"
    except ValueError as e:
        assert str(e) == "Unknown format: invalid. Use 'hex' or 'base64'."


# LLM-generated content at query #2
#--------------------------

```python
def test_Cryptographic_certificate_fingerprint():
    crypto = Cryptographic()

    # Test default algorithm (sha256)
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 2 * 32 + 31  # 32 bytes in hex with colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)

    # Test sha1 algorithm
    fingerprint_sha1 = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(fingerprint_sha1, str)
    assert len(fingerprint_sha1) == 2 * 20 + 19  # 20 bytes in hex with colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint_sha1)

    # Test invalid algorithm
    with pytest.raises(ValueError):
        crypto.certificate_fingerprint(algorithm="invalid")


# LLM-generated content at query #3
#--------------------------

```python
def test_Cryptographic_api_key():
    crypto = Cryptographic()

    # Test default parameters
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 32

    # Test with prefix
    key_with_prefix = crypto.api_key(prefix="sk_")
    assert key_with_prefix.startswith("sk_")
    assert len(key_with_prefix) == 35

    # Test with base64 format
    key_base64 = crypto.api_key(fmt="base64")
    assert isinstance(key_base64, str)
    assert len(key_base64) == 32

    # Test with prefix and base64 format
    key_base64_with_prefix = crypto.api_key(prefix="pk_", fmt="base64")
    assert key_base64_with_prefix.startswith("pk_")
    assert len(key_base64_with_prefix) == 35

    # Test with custom length
    key_custom_length = crypto.api_key(length=16)
    assert len(key_custom_length) == 16

    # Test with custom length and prefix
    key_custom_length_with_prefix = crypto.api_key(prefix="api_", length=16)
    assert key_custom_length_with_prefix.startswith("api_")
    assert len(key_custom_length_with_prefix) == 20

    # Test with custom length and base64 format
    key_custom_length_base64 = crypto.api_key(length=16, fmt="base64")
    assert len(key_custom_length_base64) == 16

    # Test with custom length, prefix, and base64 format
    key_custom_length_base64_with_prefix = crypto.api_key(prefix="api_", length=16, fmt="base64")
    assert key_custom_length_base64_with_prefix.startswith("api_")
    assert len(key_custom_length_base64_with_prefix) == 20

    # Test invalid format
    with pytest.raises(ValueError):
        crypto.api_key(fmt="invalid_format")


# LLM-generated content at query #4
#--------------------------

```python
def test_Cryptographic_certificate_fingerprint():
    crypto = Cryptographic()

    # Test default sha256
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 95  # 32 bytes * 2 (hex) + 31 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)
    assert fingerprint.count(":") == 31

    # Test sha1
    fingerprint_sha1 = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(fingerprint_sha1, str)
    assert len(fingerprint_sha1) == 119  # 20 bytes * 2 (hex) + 39 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint_sha1)
    assert fingerprint_sha1.count(":") == 39

    # Test invalid algorithm
    with pytest.raises(ValueError):
        crypto.certificate_fingerprint(algorithm="invalid")


# LLM-generated content at query #5
#--------------------------

```python
def test_Cryptographic_api_key():
    crypto = Cryptographic()

    # Test default API key (hex format, no prefix)
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 32
    assert all(c in "0123456789abcdef" for c in key)

    # Test API key with prefix
    key_with_prefix = crypto.api_key(prefix="sk_")
    assert key_with_prefix.startswith("sk_")
    assert len(key_with_prefix) == 35  # 32 + 3 (prefix length)
    assert all(c in "0123456789abcdef" for c in key_with_prefix[3:])

    # Test API key with base64 format
    key_base64 = crypto.api_key(fmt="base64")
    assert isinstance(key_base64, str)
    assert len(key_base64) == 32
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=" for c in key_base64)

    # Test API key with prefix and base64 format
    key_base64_with_prefix = crypto.api_key(prefix="pk_", fmt="base64")
    assert key_base64_with_prefix.startswith("pk_")
    assert len(key_base64_with_prefix) == 35  # 32 + 3 (prefix length)
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=" for c in key_base64_with_prefix[3:])

    # Test API key with custom length
    key_custom_length = crypto.api_key(length=16, fmt="hex")
    assert len(key_custom_length) == 16
    assert all(c in "0123456789abcdef" for c in key_custom_length)

    # Test API key with invalid format
    try:
        crypto.api_key(fmt="invalid")
        assert False, "Expected ValueError for invalid format"
    except ValueError as e:
        assert str(e) == "Unknown format: invalid. Use 'hex' or 'base64'."


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Cryptographic_certificate_fingerprint():
    crypto = Cryptographic()

    # Test default algorithm (sha256)
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 95  # 32 bytes * 2 (hex) + 31 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)
    assert fingerprint.count(":") == 31

    # Test sha1 algorithm
    fingerprint_sha1 = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(fingerprint_sha1, str)
    assert len(fingerprint_sha1) == 119  # 20 bytes * 2 (hex) + 39 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint_sha1)
    assert fingerprint_sha1.count(":") == 39

    # Test invalid algorithm
    with pytest.raises(ValueError):
        crypto.certificate_fingerprint(algorithm="invalid")


# LLM-generated content at query #2
#--------------------------

```python
def test_Cryptographic_api_key():
    crypto = Cryptographic()

    # Test default parameters
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 32

    # Test with prefix
    key_with_prefix = crypto.api_key(prefix="sk_")
    assert key_with_prefix.startswith("sk_")
    assert len(key_with_prefix) == 35

    # Test with different length
    key_long = crypto.api_key(length=64)
    assert len(key_long) == 64

    # Test with base64 format
    key_base64 = crypto.api_key(fmt="base64")
    assert isinstance(key_base64, str)
    assert len(key_base64) == 32

    # Test with prefix and base64 format
    key_base64_prefix = crypto.api_key(prefix="pk_", fmt="base64")
    assert key_base64_prefix.startswith("pk_")
    assert len(key_base64_prefix) == 35

    # Test invalid format
    with pytest.raises(ValueError):
        crypto.api_key(fmt="invalid")


# LLM-generated content at query #3
#--------------------------

```python
def test_Cryptographic_api_key():
    crypto = Cryptographic()

    # Test default parameters
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 32

    # Test with prefix
    key_with_prefix = crypto.api_key(prefix="sk_")
    assert key_with_prefix.startswith("sk_")
    assert len(key_with_prefix) == 35

    # Test with different length
    key_custom_length = crypto.api_key(length=16)
    assert len(key_custom_length) == 16

    # Test with base64 format
    key_base64 = crypto.api_key(fmt="base64")
    assert isinstance(key_base64, str)
    assert len(key_base64) == 32

    # Test with prefix and base64 format
    key_prefix_base64 = crypto.api_key(prefix="pk_", fmt="base64")
    assert key_prefix_base64.startswith("pk_")
    assert len(key_prefix_base64) == 35

    # Test invalid format
    with pytest.raises(ValueError):
        crypto.api_key(fmt="invalid")


# LLM-generated content at query #4
#--------------------------

```python
def test_Cryptographic_certificate_fingerprint():
    crypto = Cryptographic()

    # Test default algorithm (sha256)
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 95  # 32 bytes * 2 (hex) + 31 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)
    assert fingerprint.count(":") == 31

    # Test sha1 algorithm
    fingerprint_sha1 = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(fingerprint_sha1, str)
    assert len(fingerprint_sha1) == 119  # 20 bytes * 2 (hex) + 39 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint_sha1)
    assert fingerprint_sha1.count(":") == 39

    # Test invalid algorithm
    with pytest.raises(ValueError):
        crypto.certificate_fingerprint(algorithm="invalid")


# LLM-generated content at query #5
#--------------------------

```python
def test_Cryptographic_certificate_fingerprint():
    crypto = Cryptographic()

    # Test default algorithm (sha256)
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 95  # 32 bytes * 2 (hex) + 31 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)
    assert fingerprint.count(":") == 31

    # Test sha1 algorithm
    fingerprint_sha1 = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(fingerprint_sha1, str)
    assert len(fingerprint_sha1) == 119  # 20 bytes * 2 (hex) + 39 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint_sha1)
    assert fingerprint_sha1.count(":") == 39

    # Test invalid algorithm
    with pytest.raises(ValueError):
        crypto.certificate_fingerprint(algorithm="invalid")


