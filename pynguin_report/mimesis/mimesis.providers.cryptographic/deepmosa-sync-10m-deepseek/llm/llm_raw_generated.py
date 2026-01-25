####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_certificate_fingerprint_default_algorithm():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    parts = result.split(":")
    assert len(parts) == 32
    assert all(len(p) == 2 for p in parts)
    assert all(c in "0123456789ABCDEF" for p in parts for c in p)

def test_certificate_fingerprint_sha256():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint("sha256")
    parts = result.split(":")
    assert len(parts) == 32
    assert all(len(p) == 2 for p in parts)
    assert all(c in "0123456789ABCDEF" for p in parts for c in p)

def test_certificate_fingerprint_sha1():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint("sha1")
    parts = result.split(":")
    assert len(parts) == 20
    assert all(len(p) == 2 for p in parts)
    assert all(c in "0123456789ABCDEF" for p in parts for c in p)

def test_certificate_fingerprint_uppercase():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    assert result == result.upper()

def test_certificate_fingerprint_invalid_algorithm():
    crypto = Cryptographic()
    try:
        crypto.certificate_fingerprint("md5")
        assert False
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


# LLM-generated content at query #2
#--------------------------

def test_api_key_default():
    crypto = Cryptographic()
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 32

def test_api_key_with_prefix():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="sk_")
    assert isinstance(key, str)
    assert key.startswith("sk_")
    assert len(key) == 32 + len("sk_")

def test_api_key_with_custom_length():
    crypto = Cryptographic()
    key = crypto.api_key(length=64)
    assert isinstance(key, str)
    assert len(key) == 64

def test_api_key_with_prefix_and_length():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="api_", length=48)
    assert isinstance(key, str)
    assert key.startswith("api_")
    assert len(key) == 48 + len("api_")

def test_api_key_format_hex():
    crypto = Cryptographic()
    key = crypto.api_key(fmt="hex")
    assert isinstance(key, str)
    assert all(c in "0123456789abcdef" for c in key)

def test_api_key_format_base64():
    crypto = Cryptographic()
    key = crypto.api_key(fmt="base64")
    assert isinstance(key, str)
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in key)

def test_api_key_format_base64_with_prefix():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="pk_", fmt="base64")
    assert isinstance(key, str)
    assert key.startswith("pk_")
    key_part = key[len("pk_"):]
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in key_part)

def test_api_key_invalid_format():
    crypto = Cryptographic()
    try:
        crypto.api_key(fmt="invalid")
        assert False
    except ValueError as e:
        assert "Unknown format" in str(e)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_certificate_fingerprint_default_algorithm():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    parts = result.split(":")
    assert len(parts) == 32
    assert all(len(p) == 2 for p in parts)
    assert all(c in "0123456789ABCDEF" for p in parts for c in p)

def test_certificate_fingerprint_sha256():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint(algorithm="sha256")
    parts = result.split(":")
    assert len(parts) == 32
    assert all(len(p) == 2 for p in parts)
    assert all(c in "0123456789ABCDEF" for p in parts for c in p)

def test_certificate_fingerprint_sha1():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint(algorithm="sha1")
    parts = result.split(":")
    assert len(parts) == 20
    assert all(len(p) == 2 for p in parts)
    assert all(c in "0123456789ABCDEF" for p in parts for c in p)

def test_certificate_fingerprint_invalid_algorithm():
    crypto = Cryptographic()
    try:
        crypto.certificate_fingerprint(algorithm="md5")
        assert False
    except ValueError as e:
        assert "Unknown algorithm" in str(e)

def test_certificate_fingerprint_uppercase_output():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    assert result.isupper()

def test_certificate_fingerprint_colon_separated():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    assert ":" in result
    assert result.count(":") == 31

def test_certificate_fingerprint_sha1_colon_separated():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint(algorithm="sha1")
    assert ":" in result
    assert result.count(":") == 19

def test_certificate_fingerprint_hex_characters_only():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    clean = result.replace(":", "")
    assert all(c in "0123456789ABCDEF" for c in clean)

def test_certificate_fingerprint_different_calls_produce_different_results():
    crypto = Cryptographic()
    result1 = crypto.certificate_fingerprint()
    result2 = crypto.certificate_fingerprint()
    assert result1 != result2

def test_certificate_fingerprint_sha1_different_calls_produce_different_results():
    crypto = Cryptographic()
    result1 = crypto.certificate_fingerprint(algorithm="sha1")
    result2 = crypto.certificate_fingerprint(algorithm="sha1")
    assert result1 != result2


# LLM-generated content at query #2
#--------------------------

def test_api_key_default():
    crypto = Cryptographic()
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 64

def test_api_key_with_prefix():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="sk_")
    assert isinstance(key, str)
    assert key.startswith("sk_")
    assert len(key) == 67

def test_api_key_with_length():
    crypto = Cryptographic()
    key = crypto.api_key(length=16)
    assert isinstance(key, str)
    assert len(key) == 32

def test_api_key_with_prefix_and_length():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="api_", length=24)
    assert isinstance(key, str)
    assert key.startswith("api_")
    assert len(key) == 52

def test_api_key_format_hex():
    crypto = Cryptographic()
    key = crypto.api_key(fmt="hex")
    assert isinstance(key, str)
    assert all(c in "0123456789abcdef" for c in key)

def test_api_key_format_base64():
    crypto = Cryptographic()
    key = crypto.api_key(fmt="base64")
    assert isinstance(key, str)
    assert len(key) == 32

def test_api_key_format_base64_with_prefix():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="pk_", fmt="base64")
    assert isinstance(key, str)
    assert key.startswith("pk_")
    assert len(key) == 35

def test_api_key_format_base64_with_length():
    crypto = Cryptographic()
    key = crypto.api_key(length=48, fmt="base64")
    assert isinstance(key, str)
    assert len(key) == 48

def test_api_key_invalid_format():
    crypto = Cryptographic()
    try:
        crypto.api_key(fmt="invalid")
        assert False
    except ValueError as e:
        assert "Unknown format" in str(e)


