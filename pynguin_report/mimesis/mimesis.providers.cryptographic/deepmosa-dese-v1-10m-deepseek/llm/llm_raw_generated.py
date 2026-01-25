####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_api_key_default():
    crypto = Cryptographic()
    key = crypto.api_key()
    assert len(key) == 32
    assert all(c in "0123456789abcdef" for c in key)

def test_api_key_with_prefix():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="sk_")
    assert key.startswith("sk_")
    assert len(key) == 35  # 32 + 3 for prefix

def test_api_key_base64_format():
    crypto = Cryptographic()
    key = crypto.api_key(fmt="base64")
    assert len(key) == 32
    assert all(c.isalnum() or c in "-_" for c in key)

def test_api_key_with_prefix_and_base64():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="pk_", fmt="base64")
    assert key.startswith("pk_")
    assert len(key) == 35  # 32 + 3 for prefix

def test_api_key_custom_length():
    crypto = Cryptographic()
    key = crypto.api_key(length=64)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)

def test_api_key_invalid_format_raises_error():
    crypto = Cryptographic()
    try:
        crypto.api_key(fmt="invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_certificate_fingerprint_default_algorithm():
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint()
    assert len(fingerprint) == 95  # 32 bytes * 2 + 31 colons
    assert all(part.isalnum() and len(part) == 2 for part in fingerprint.split(":"))
    assert fingerprint == fingerprint.upper()

def test_certificate_fingerprint_sha256():
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint("sha256")
    assert len(fingerprint) == 95  # 32 bytes * 2 + 31 colons
    assert all(part.isalnum() and len(part) == 2 for part in fingerprint.split(":"))
    assert fingerprint == fingerprint.upper()

def test_certificate_fingerprint_sha1():
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint("sha1")
    assert len(fingerprint) == 59  # 20 bytes * 2 + 19 colons
    assert all(part.isalnum() and len(part) == 2 for part in fingerprint.split(":"))
    assert fingerprint == fingerprint.upper()

def test_certificate_fingerprint_invalid_algorithm():
    crypto = Cryptographic()
    try:
        crypto.certificate_fingerprint("md5")
        assert False  # Should raise ValueError
    except ValueError:
        pass


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_api_key_with_default_params():
    crypto = Cryptographic()
    key = crypto.api_key()
    assert len(key) == 32
    assert all(c in '0123456789abcdef' for c in key)

def test_api_key_with_prefix():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="sk_")
    assert len(key) == 35
    assert key.startswith("sk_")
    assert all(c in '0123456789abcdef' for c in key[3:])

def test_api_key_with_base64_format():
    crypto = Cryptographic()
    key = crypto.api_key(fmt="base64")
    assert len(key) == 32
    assert all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_' for c in key)

def test_api_key_with_prefix_and_base64_format():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="pk_", fmt="base64")
    assert len(key) == 35
    assert key.startswith("pk_")
    assert all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_' for c in key[3:])

def test_api_key_with_custom_length():
    crypto = Cryptographic()
    key = crypto.api_key(length=16)
    assert len(key) == 16
    assert all(c in '0123456789abcdef' for c in key)

def test_api_key_with_invalid_format():
    crypto = Cryptographic()
    try:
        crypto.api_key(fmt="invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_certificate_fingerprint_sha256():
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint("sha256")
    assert len(fingerprint) == 95  # 32 bytes * 2 chars + 31 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)
    assert fingerprint.count(":") == 31

def test_certificate_fingerprint_sha1():
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint("sha1")
    assert len(fingerprint) == 59  # 20 bytes * 2 chars + 19 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)
    assert fingerprint.count(":") == 19

def test_certificate_fingerprint_invalid_algorithm():
    crypto = Cryptographic()
    try:
        crypto.certificate_fingerprint("invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass


