####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_api_key_default():
    crypto = Cryptographic()
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 32
    assert all(c in "0123456789abcdef" for c in key)

def test_api_key_with_prefix():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="sk_")
    assert isinstance(key, str)
    assert key.startswith("sk_")
    assert len(key) == 35
    assert all(c in "0123456789abcdef" for c in key[3:])

def test_api_key_with_length():
    crypto = Cryptographic()
    key = crypto.api_key(length=64)
    assert isinstance(key, str)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)

def test_api_key_with_prefix_and_length():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="api_", length=48)
    assert isinstance(key, str)
    assert key.startswith("api_")
    assert len(key) == 52
    assert all(c in "0123456789abcdef" for c in key[4:])

def test_api_key_format_hex():
    crypto = Cryptographic()
    key = crypto.api_key(fmt="hex")
    assert isinstance(key, str)
    assert len(key) == 32
    assert all(c in "0123456789abcdef" for c in key)

def test_api_key_format_base64():
    crypto = Cryptographic()
    key = crypto.api_key(fmt="base64")
    assert isinstance(key, str)
    assert len(key) == 32
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in key)

def test_api_key_format_base64_with_prefix():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="pk_", fmt="base64")
    assert isinstance(key, str)
    assert key.startswith("pk_")
    assert len(key) == 35
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in key[3:])

def test_api_key_format_base64_with_length():
    crypto = Cryptographic()
    key = crypto.api_key(length=64, fmt="base64")
    assert isinstance(key, str)
    assert len(key) == 64
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in key)

def test_api_key_invalid_format():
    crypto = Cryptographic()
    try:
        crypto.api_key(fmt="invalid")
        assert False
    except ValueError as e:
        assert str(e) == "Unknown format: invalid. Use 'hex' or 'base64'."

def test_api_key_empty_prefix():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="")
    assert isinstance(key, str)
    assert len(key) == 32
    assert all(c in "0123456789abcdef" for c in key)

def test_api_key_length_zero():
    crypto = Cryptographic()
    key = crypto.api_key(length=0)
    assert isinstance(key, str)
    assert len(key) == 0

def test_api_key_length_zero_with_prefix():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="test_", length=0)
    assert isinstance(key, str)
    assert key == "test_"

def test_api_key_length_odd():
    crypto = Cryptographic()
    key = crypto.api_key(length=31)
    assert isinstance(key, str)
    assert len(key) == 31
    assert all(c in "0123456789abcdef" for c in key)

def test_api_key_length_odd_with_prefix():
    crypto = Cryptographic()
    key = crypto.api_key(prefix="odd_", length=31)
    assert isinstance(key, str)
    assert key.startswith("odd_")
    assert len(key) == 35
    assert all(c in "0123456789abcdef" for c in key[4:])


# LLM-generated content at query #2
#--------------------------

def test_certificate_fingerprint_default_algorithm():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    parts = result.split(":")
    assert len(parts) == 32
    assert all(len(part) == 2 for part in parts)
    assert all(c in "0123456789ABCDEF" for part in parts for c in part)
    assert result == result.upper()

def test_certificate_fingerprint_sha256():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint(algorithm="sha256")
    parts = result.split(":")
    assert len(parts) == 32
    assert all(len(part) == 2 for part in parts)
    assert all(c in "0123456789ABCDEF" for part in parts for c in part)
    assert result == result.upper()

def test_certificate_fingerprint_sha1():
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint(algorithm="sha1")
    parts = result.split(":")
    assert len(parts) == 20
    assert all(len(part) == 2 for part in parts)
    assert all(c in "0123456789ABCDEF" for part in parts for c in part)
    assert result == result.upper()

def test_certificate_fingerprint_unsupported_algorithm():
    crypto = Cryptographic()
    try:
        crypto.certificate_fingerprint(algorithm="md5")
        assert False
    except ValueError as e:
        assert str(e) == "Unknown algorithm: md5. Use 'sha256' or 'sha1'."

def test_certificate_fingerprint_format_consistency():
    crypto = Cryptographic()
    result1 = crypto.certificate_fingerprint()
    result2 = crypto.certificate_fingerprint()
    assert result1 != result2
    parts1 = result1.split(":")
    parts2 = result2.split(":")
    assert len(parts1) == len(parts2)
    assert all(len(p) == 2 for p in parts1)
    assert all(len(p) == 2 for p in parts2)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_api_key_default():
    crypto = Cryptographic()
    result = crypto.api_key()
    assert isinstance(result, str)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)

def test_api_key_with_prefix():
    crypto = Cryptographic()
    result = crypto.api_key(prefix="sk_")
    assert isinstance(result, str)
    assert result.startswith("sk_")
    assert len(result) == 64 + 3
    assert all(c in "0123456789abcdef" for c in result[3:])

def test_api_key_with_custom_length():
    crypto = Cryptographic()
    result = crypto.api_key(length=16)
    assert isinstance(result, str)
    assert len(result) == 32
    assert all(c in "0123456789abcdef" for c in result)

def test_api_key_base64_format():
    crypto = Cryptographic()
    result = crypto.api_key(fmt="base64")
    assert isinstance(result, str)
    assert len(result) == 32
    try:
        urlsafe_b64decode(result + "=" * (4 - len(result) % 4))
        assert True
    except Exception:
        assert False

def test_api_key_base64_with_prefix():
    crypto = Cryptographic()
    result = crypto.api_key(prefix="pk_", fmt="base64")
    assert isinstance(result, str)
    assert result.startswith("pk_")
    assert len(result) == 32 + 3
    try:
        urlsafe_b64decode(result[3:] + "=" * (4 - len(result[3:]) % 4))
        assert True
    except Exception:
        assert False

def test_api_key_raises_value_error_for_invalid_format():
    crypto = Cryptographic()
    try:
        crypto.api_key(fmt="invalid")
        assert False
    except ValueError as e:
        assert "Unknown format" in str(e)


# LLM-generated content at query #2
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

def test_certificate_fingerprint_different_calls_produce_different_results():
    crypto = Cryptographic()
    result1 = crypto.certificate_fingerprint()
    result2 = crypto.certificate_fingerprint()
    assert result1 != result2


