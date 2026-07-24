####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Signer_unsign():
    # Test basic unsign
    signer = Signer("secret-key")
    signed = signer.sign(b"test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"

    # Test unsign with string input
    signed_str = signer.sign("test-value")
    result = signer.unsign(signed_str)
    assert result == b"test-value"

    # Test unsign with key rotation - verify with older key
    signer_old = Signer("old-secret")
    signed_old = signer_old.sign(b"value-signed-with-old-key")
    signer_new = Signer(["new-secret", "old-secret"])
    result = signer_new.unsign(signed_old)
    assert result == b"value-signed-with-old-key"

    # Test unsign raises BadSignature when no separator found
    try:
        signer.unsign(b"no-separator")
        assert False, "Should have raised BadSignature"
    except BadSignature as e:
        assert "no '.' found" in str(e)

    # Test unsign raises BadSignature when signature doesn't match
    try:
        signer.unsign(b"value.invalid-signature")
        assert False, "Should have raised BadSignature"
    except BadSignature as e:
        assert "does not match" in str(e)

    # Test unsign with different separator
    signer_with_sep = Signer("secret-key", sep=b"|")
    signed = signer_with_sep.sign(b"test-value")
    result = signer_with_sep.unsign(signed)
    assert result == b"test-value"

    # Test unsign fails with wrong separator
    try:
        signer.unsign(signed)
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test unsign with NoneAlgorithm
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    signed = none_signer.sign(b"test-value")
    result = none_signer.unsign(signed)
    assert result == b"test-value"

    # Test unsign with different key derivation methods
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=derivation)
        signed = signer.sign(b"test-value")
        result = signer.unsign(signed)
        assert result == b"test-value"

    # Test unsign with empty value
    signed = signer.sign(b"")
    result = signer.unsign(signed)
    assert result == b""

    # Test unsign preserves value with special characters
    special_value = b"value-with-special-chars:./\\"
    signed = signer.sign(special_value)
    result = signer.unsign(signed)
    assert result == special_value
```


# LLM-generated content at query #2
#--------------------------

```python
def test_Signer_unsign():
    # Test with default parameters
    signer = Signer("secret-key")
    signed = signer.sign("test-value")
    assert signer.unsign(signed) == b"test-value"

    # Test with custom separator
    signer = Signer("secret-key", sep=b"|")
    signed = signer.sign("test-value")
    assert signer.unsign(signed) == b"test-value"

    # Test with bytes input
    signer = Signer(b"secret-key")
    signed = signer.sign(b"test-value")
    assert signer.unsign(signed) == b"test-value"

    # Test with salt
    signer = Signer("secret-key", salt="custom-salt")
    signed = signer.sign("test-value")
    assert signer.unsign(signed) == b"test-value"

    # Test with key rotation (multiple secret keys)
    signer = Signer(["old-key", "new-key"])
    signed = signer.sign("test-value")
    assert signer.unsign(signed) == b"test-value"

    # Test with key derivation methods
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=derivation)
        signed = signer.sign("test-value")
        assert signer.unsign(signed) == b"test-value"

    # Test with digest method
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    signed = signer.sign("test-value")
    assert signer.unsign(signed) == b"test-value"

    # Test with NoneAlgorithm
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    signed = signer.sign("test-value")
    assert signer.unsign(signed) == b"test-value"

    # Test raises BadSignature when no separator found
    signer = Signer("secret-key")
    try:
        signer.unsign(b"no-separator")
        assert False, "Should have raised BadSignature"
    except BadSignature as e:
        assert "No '.' found in value" in str(e)

    # Test raises BadSignature when signature doesn't match
    signer = Signer("secret-key")
    try:
        signer.unsign(b"value.invalid-signature")
        assert False, "Should have raised BadSignature"
    except BadSignature as e:
        assert "does not match" in str(e)

    # Test with empty value
    signer = Signer("secret-key")
    signed = signer.sign(b"")
    assert signer.unsign(signed) == b""

    # Test with special characters in value
    signer = Signer("secret-key")
    signed = signer.sign("value with spaces and !@#$%")
    assert signer.unsign(signed) == b"value with spaces and !@#$%"

    # Test with Unicode string
    signer = Signer("secret-key")
    signed = signer.sign("café")
    assert signer.unsign(signed) == "café".encode()

    # Test with long value
    signer = Signer("secret-key")
    long_value = "a" * 10000
    signed = signer.sign(long_value)
    assert signer.unsign(signed) == long_value.encode()

    # Test verify_signature returns False for invalid signature
    signer = Signer("secret-key")
    assert not signer.verify_signature(b"value", b"invalid-sig")

    # Test unsign with different key derivation
    signer = Signer("secret-key", key_derivation="hmac")
    signed = signer.sign("test-value")
    assert signer.unsign(signed) == b"test-value"

    # Test unsign with NoneAlgorithm and empty signature
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    # With NoneAlgorithm, any signature should verify
    assert signer.unsign(b"value.") == b"value"
    assert signer.unsign(b"value.anything") == b"value"

    # Test that separator cannot be in base64 alphabet
    try:
        Signer("secret-key", sep=b"a")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot be used" in str(e)

    # Test with integer secret_key (should fail if not str/bytes/iterable)
    try:
        Signer(12345)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test validate method returns True for valid signed value
    signer = Signer("secret-key")
    signed = signer.sign("test-value")
    assert signer.validate(signed)

    # Test validate method returns False for invalid signed value
    assert not signer.validate(b"invalid-value")


# LLM-generated content at query #3
#--------------------------

```python
def test_Signer_derive_key():
    signer = Signer("test-secret", salt="test-salt")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) > 0

def test_Signer_derive_key_specific_secret():
    signer = Signer("test-secret", salt="test-salt")
    key = signer.derive_key("custom-secret")
    assert isinstance(key, bytes)
    assert len(key) > 0

def test_Signer_derive_key_concat():
    signer = Signer("test-secret", salt="test-salt", key_derivation="concat")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) == 20  # SHA1 digest size

def test_Signer_derive_key_django_concat():
    signer = Signer("test-secret", salt="test-salt", key_derivation="django-concat")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) == 20  # SHA1 digest size

def test_Signer_derive_key_hmac():
    signer = Signer("test-secret", salt="test-salt", key_derivation="hmac")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) == 20  # SHA1 digest size

def test_Signer_derive_key_none():
    signer = Signer("test-secret", salt="test-salt", key_derivation="none")
    key = signer.derive_key()
    assert key == b"test-secret"

def test_Signer_derive_key_invalid_derivation():
    signer = Signer("test-secret", salt="test-salt", key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer.derive_key()

def test_Signer_derive_key_different_secret_keys():
    signer = Signer(["old-secret", "new-secret"], salt="test-salt")
    key_default = signer.derive_key()
    key_old = signer.derive_key("old-secret")
    key_new = signer.derive_key("new-secret")
    assert key_default == key_new
    assert key_old != key_new

def test_Signer_derive_key_with_bytes_secret():
    signer = Signer(b"test-secret", salt="test-salt")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) > 0

def test_Signer_derive_key_consistency():
    signer1 = Signer("test-secret", salt="test-salt")
    signer2 = Signer("test-secret", salt="test-salt")
    assert signer1.derive_key() == signer2.derive_key()
```


# LLM-generated content at query #4
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with a simple secret key and value
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with multiple secret keys (key rotation)
    signer_rotation = Signer(["old-key", "new-key"])
    value = b"test-rotation"
    sig = signer_rotation.get_signature(value)
    # Verify with the newest key
    assert signer_rotation.verify_signature(value, sig) is True

    # Test with wrong signature
    wrong_sig = base64_encode(b"wrong-signature")
    assert signer.verify_signature(value, wrong_sig) is False

    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, "invalid-base64!!") is False

    # Test with HMAC algorithm
    hmac_signer = Signer("hmac-key", algorithm=HMACAlgorithm(hashlib.sha256))
    value = b"hmac-test"
    sig = hmac_signer.get_signature(value)
    assert hmac_signer.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    none_signer = Signer("none-key", algorithm=NoneAlgorithm())
    value = b"none-test"
    sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, sig) is True
    assert none_signer.verify_signature(value, b"") is True  # Empty signature should match

    # Test with different key derivation methods
    concat_signer = Signer("concat-key", key_derivation="concat")
    value = b"concat-test"
    sig = concat_signer.get_signature(value)
    assert concat_signer.verify_signature(value, sig) is True

    hmac_derived_signer = Signer("hmac-derived-key", key_derivation="hmac")
    value = b"hmac-derived-test"
    sig = hmac_derived_signer.get_signature(value)
    assert hmac_derived_signer.verify_signature(value, sig) is True

    # Test with string value
    string_signer = Signer("string-key")
    sig = string_signer.get_signature("string-value")
    assert string_signer.verify_signature("string-value", sig) is True

    # Test with custom salt
    salt_signer = Signer("salt-key", salt=b"custom-salt")
    value = b"salt-test"
    sig = salt_signer.get_signature(value)
    assert salt_signer.verify_signature(value, sig) is True

    # Test with custom separator
    sep_signer = Signer("sep-key", sep=b"|")
    value = b"sep-test"
    sig = sep_signer.get_signature(value)
    assert sep_signer.verify_signature(value, sig) is True

    # Test with None salt (should use default)
    none_salt_signer = Signer("none-salt-key", salt=None)
    value = b"none-salt-test"
    sig = none_salt_signer.get_signature(value)
    assert none_salt_signer.verify_signature(value, sig) is True

    # Test with integer-like bytes value
    int_signer = Signer("int-key")
    value = b"12345"
    sig = int_signer.get_signature(value)
    assert int_signer.verify_signature(value, sig) is True
```


# LLM-generated content at query #5
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret", salt="salt")
    key = signer.derive_key()
    expected = hashlib.sha1(b"salt" + b"signer" + b"secret").digest()
    assert key == expected
    
    # Test with concat key derivation
    signer = Signer("secret", salt="salt", key_derivation="concat")
    key = signer.derive_key()
    expected = hashlib.sha1(b"salt" + b"secret").digest()
    assert key == expected
    
    # Test with hmac key derivation
    signer = Signer("secret", salt="salt", key_derivation="hmac")
    key = signer.derive_key()
    mac = hmac.new(b"secret", digestmod=hashlib.sha1)
    mac.update(b"salt")
    expected = mac.digest()
    assert key == expected
    
    # Test with none key derivation
    signer = Signer("secret", salt="salt", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret"
    
    # Test with explicit secret_key parameter
    signer = Signer("secret", salt="salt")
    key = signer.derive_key("other_secret")
    expected = hashlib.sha1(b"salt" + b"signer" + b"other_secret").digest()
    assert key == expected
    
    # Test with bytes secret key
    signer = Signer(b"secret", salt=b"salt")
    key = signer.derive_key()
    expected = hashlib.sha1(b"salt" + b"signer" + b"secret").digest()
    assert key == expected
    
    # Test with multiple secret keys (uses last one by default)
    signer = Signer(["key1", "key2", "key3"], salt="salt")
    key = signer.derive_key()
    expected = hashlib.sha1(b"salt" + b"signer" + b"key3").digest()
    assert key == expected
    
    # Test with custom digest method
    signer = Signer("secret", salt="salt", digest_method=hashlib.sha256)
    key = signer.derive_key()
    expected = hashlib.sha256(b"salt" + b"signer" + b"secret").digest()
    assert key == expected
    
    # Test that derive_key returns bytes
    signer = Signer("secret", salt="salt")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    
    # Test that unknown key derivation raises TypeError
    signer = Signer("secret", salt="salt", key_derivation="unknown")
    import pytest
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer.derive_key()
```


# LLM-generated content at query #6
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    
    # Valid signature should return True
    assert signer.verify_signature(value, sig) == True
    
    # Invalid signature should return False
    assert signer.verify_signature(value, b"invalid-sig") == False
    
    # Test with different value
    assert signer.verify_signature(b"different-value", sig) == False
    
    # Test with string value
    assert signer.verify_signature("test-value", sig) == True
    
    # Test with key rotation (multiple secret keys)
    signer_rotated = Signer(["old-key", "new-key"])
    value2 = b"test-value-2"
    sig2 = signer_rotated.get_signature(value2)
    
    # Should verify with the newest key
    assert signer_rotated.verify_signature(value2, sig2) == True
    
    # Should also verify with old key if signature was created with it
    old_signer = Signer("old-key")
    old_sig = old_signer.get_signature(value2)
    assert signer_rotated.verify_signature(value2, old_sig) == True
    
    # Test with NoneAlgorithm
    none_signer = Signer("key", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) == True
    
    # Test with custom salt
    custom_salt_signer = Signer("key", salt="custom-salt")
    custom_sig = custom_salt_signer.get_signature(value)
    assert custom_salt_signer.verify_signature(value, custom_sig) == True
    
    # Different salt should not verify
    different_salt_signer = Signer("key", salt="different-salt")
    assert different_salt_signer.verify_signature(value, custom_sig) == False
    
    # Test with invalid base64 signature (should return False)
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False
    
    # Test with empty value
    empty_sig = signer.get_signature(b"")
    assert signer.verify_signature(b"", empty_sig) == True
    
    # Test with custom separator
    custom_sep_signer = Signer("key", sep=b"|")
    custom_sep_sig = custom_sep_signer.get_signature(value)
    assert custom_sep_signer.verify_signature(value, custom_sep_sig) == True
```


# LLM-generated content at query #7
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with a basic signer
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    
    # Valid signature should return True
    assert signer.verify_signature(value, sig) is True
    
    # Invalid signature should return False
    assert signer.verify_signature(value, b"invalid-sig") is False
    
    # Invalid base64 signature should return False
    assert signer.verify_signature(value, b"!!!not-base64!!!") is False
    
    # Test with key rotation
    signer2 = Signer(["old-key", "new-key"])
    value2 = b"test-value-2"
    sig2 = signer2.get_signature(value2)
    
    # Should verify with both old and new keys
    assert signer2.verify_signature(value2, sig2) is True
    
    # Test with different salt
    signer3 = Signer("secret-key", salt=b"custom-salt")
    value3 = b"test-value-3"
    sig3 = signer3.get_signature(value3)
    
    # Should verify with custom salt
    assert signer3.verify_signature(value3, sig3) is True
    
    # Different salt should not verify
    signer4 = Signer("secret-key", salt=b"different-salt")
    assert signer4.verify_signature(value3, sig3) is False
    
    # Test with HMAC key derivation
    signer5 = Signer("secret-key", key_derivation="hmac")
    value5 = b"test-value-5"
    sig5 = signer5.get_signature(value5)
    assert signer5.verify_signature(value5, sig5) is True
    
    # Test with empty value
    signer6 = Signer("secret-key")
    empty_value = b""
    sig6 = signer6.get_signature(empty_value)
    assert signer6.verify_signature(empty_value, sig6) is True
    
    # Test with string input (not bytes)
    signer7 = Signer("secret-key")
    str_value = "test-string"
    sig7 = signer7.get_signature(str_value)
    assert signer7.verify_signature(str_value, sig7) is True
```


# LLM-generated content at query #8
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with a simple string value
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with string value (not bytes)
    value_str = "test-value"
    sig_str = signer.get_signature(value_str)
    assert signer.verify_signature(value_str, sig_str) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) is True

    # Test with corrupted base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False

    # Test with NoneAlgorithm (no signature)
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value_none = b"test-value"
    sig_none = b""
    assert signer_none.verify_signature(value_none, sig_none) is True

    # Test with key rotation - multiple keys
    signer_rotation = Signer(["old-key", "new-key"])
    value_rotation = b"test-value"
    sig_rotation = signer_rotation.get_signature(value_rotation)
    assert signer_rotation.verify_signature(value_rotation, sig_rotation) is True

    # Test that old keys can verify signatures (key rotation)
    old_sig = Signer("old-key").get_signature(value_rotation)
    assert signer_rotation.verify_signature(value_rotation, old_sig) is True

    # Test with different salt
    signer_salt = Signer("secret-key", salt="custom-salt")
    value_salt = b"test-value"
    sig_salt = signer_salt.get_signature(value_salt)
    assert signer_salt.verify_signature(value_salt, sig_salt) is True

    # Test that signatures with different salts don't match
    signer_diff_salt = Signer("secret-key", salt="different-salt")
    assert signer_diff_salt.verify_signature(value_salt, sig_salt) is False

    # Test with HMAC key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    value_hmac = b"test-value"
    sig_hmac = signer_hmac.get_signature(value_hmac)
    assert signer_hmac.verify_signature(value_hmac, sig_hmac) is True

    # Test with concat key derivation
    signer_concat = Signer("secret-key", key_derivation="concat")
    value_concat = b"test-value"
    sig_concat = signer_concat.get_signature(value_concat)
    assert signer_concat.verify_signature(value_concat, sig_concat) is True

    # Test with SHA256 digest method
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value_sha256 = b"test-value"
    sig_sha256 = signer_sha256.get_signature(value_sha256)
    assert signer_sha256.verify_signature(value_sha256, sig_sha256) is True

    # Test with binary data containing special characters
    binary_value = bytes(range(256))
    sig_binary = signer.get_signature(binary_value)
    assert signer.verify_signature(binary_value, sig_binary) is True

    # Test with unicode string value
    unicode_value = "héllo wörld 🎉"
    sig_unicode = signer.get_signature(unicode_value)
    assert signer.verify_signature(unicode_value, sig_unicode) is True

    # Test that empty bytes signature doesn't match for HMAC algorithm
    assert signer.verify_signature(value, b"") is False
```


# LLM-generated content at query #9
#--------------------------

```python
def test_Signer_verify_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    
    assert signer.verify_signature(value, sig) == True
    
    assert signer.verify_signature(value, b"invalid-signature") == False
    
    assert signer.verify_signature(b"different-value", sig) == False
    
    signer_rotation = Signer(["old-key", "new-key"])
    value2 = b"rotation-test"
    sig2 = signer_rotation.get_signature(value2)
    assert signer_rotation.verify_signature(value2, sig2) == True
    
    different_salt_signer = Signer("secret-key", salt=b"different-salt")
    assert different_salt_signer.verify_signature(value, sig) == False
    
    assert signer.verify_signature(value, b"") == False
    assert signer.verify_signature(value, b"!!!invalid_base64!!!") == False
    
    none_alg_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value3 = b"no-signature"
    sig3 = none_alg_signer.get_signature(value3)
    assert none_alg_signer.verify_signature(value3, sig3) == True
    
    signer_bytes_sep = Signer("secret-key", sep=b"|")
    value4 = b"custom-sep"
    sig4 = signer_bytes_sep.get_signature(value4)
    assert signer_bytes_sep.verify_signature(value4, sig4) == True
```


# LLM-generated content at query #10
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid_sig") is False

    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid_base64!!!") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value_none = b"test"
    sig_none = signer_none.get_signature(value_none)
    assert signer_none.verify_signature(value_none, sig_none) is True

    # Test with key rotation (multiple secret keys)
    signer_rotation = Signer(["old-secret", "new-secret"], salt="rotation-test")
    value_rot = b"rotation value"
    sig_rot = signer_rotation.get_signature(value_rot)
    # Should verify with both old and new keys
    assert signer_rotation.verify_signature(value_rot, sig_rot) is True

    # Test with string value
    str_value = "string value"
    sig_str = signer.get_signature(str_value)
    assert signer.verify_signature(str_value, sig_str) is True

    # Test with modified value
    modified_value = b"modified value"
    assert signer.verify_signature(modified_value, sig) is False
```


# LLM-generated content at query #11
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("test-secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid-signature")
    assert signer.verify_signature(value, invalid_sig) is False
    
    # Test with empty value
    empty_value = b""
    empty_sig = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, empty_sig) is True
    
    # Test with string input
    str_value = "test-string"
    str_sig = signer.get_signature(str_value)
    assert signer.verify_signature(str_value, str_sig) is True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, "not-base64!") is False
    
    # Test with different keys in rotation
    signer2 = Signer(["old-key", "new-key"])
    value2 = b"rotated-value"
    sig2 = signer2.get_signature(value2)
    assert signer2.verify_signature(value2, sig2) is True
    
    # Test with NoneAlgorithm
    none_signer = Signer("test", algorithm=NoneAlgorithm())
    value3 = b"none-algorithm"
    sig3 = none_signer.get_signature(value3)
    assert sig3 == base64_encode(b"")
    assert none_signer.verify_signature(value3, sig3) is True
    
    # Test with HMACAlgorithm
    hmac_signer = Signer("test", algorithm=HMACAlgorithm())
    value4 = b"hmac-test"
    sig4 = hmac_signer.get_signature(value4)
    assert hmac_signer.verify_signature(value4, sig4) is True
    
    # Test with different digest method
    sha256_signer = Signer("test", digest_method=hashlib.sha256)
    value5 = b"sha256-test"
    sig5 = sha256_signer.get_signature(value5)
    assert sha256_signer.verify_signature(value5, sig5) is True
    
    # Test with bytes separator
    sep_signer = Signer("test", sep=b"|")
    value6 = b"separator-test"
    sig6 = sep_signer.get_signature(value6)
    assert sep_signer.verify_signature(value6, sig6) is True
    
    # Test with salt=None
    no_salt_signer = Signer("test", salt=None)
    value7 = b"no-salt"
    sig7 = no_salt_signer.get_signature(value7)
    assert no_salt_signer.verify_signature(value7, sig7) is True
    
    # Test with explicit salt
    custom_salt_signer = Signer("test", salt=b"custom-salt")
    value8 = b"custom-salt-value"
    sig8 = custom_salt_signer.get_signature(value8)
    assert custom_salt_signer.verify_signature(value8, sig8) is True
    
    # Test with key_derivation="concat"
    concat_signer = Signer("test", key_derivation="concat")
    value9 = b"concat-test"
    sig9 = concat_signer.get_signature(value9)
    assert concat_signer.verify_signature(value9, sig9) is True
    
    # Test with key_derivation="hmac"
    hmac_derived_signer = Signer("test", key_derivation="hmac")
    value10 = b"hmac-derived"
    sig10 = hmac_derived_signer.get_signature(value10)
    assert hmac_derived_signer.verify_signature(value10, sig10) is True
    
    # Test with key_derivation="none"
    none_derived_signer = Signer("test", key_derivation="none")
    value11 = b"none-derived"
    sig11 = none_derived_signer.get_signature(value11)
    assert none_derived_signer.verify_signature(value11, sig11) is True
```


# LLM-generated content at query #12
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    
    # Test valid signature
    assert signer.verify_signature(value, sig) == True
    
    # Test invalid signature
    assert signer.verify_signature(value, b"invalid-sig") == False
    
    # Test with string value
    assert signer.verify_signature("test-value", sig) == True
    
    # Test with string signature
    sig_str = sig.decode("ascii")
    assert signer.verify_signature(value, sig_str) == True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False
    
    # Test with modified value
    modified_value = b"modified-value"
    assert signer.verify_signature(modified_value, sig) == False
    
    # Test with key rotation (multiple secret keys)
    signer_rotated = Signer(["old-key", "new-key"])
    value_rotated = b"test-rotated"
    
    # Sign with newest key
    sig_rotated = signer_rotated.get_signature(value_rotated)
    assert signer_rotated.verify_signature(value_rotated, sig_rotated) == True
    
    # Verify with old key should also work (since we iterate through all keys)
    old_signer = Signer("old-key")
    old_sig = old_signer.get_signature(value_rotated)
    assert signer_rotated.verify_signature(value_rotated, old_sig) == True
    
    # Test with different salt
    signer_diff_salt = Signer("secret-key", salt="different-salt")
    sig_diff_salt = signer_diff_salt.get_signature(value)
    
    # Signature from different salt should not verify with original signer
    assert signer.verify_signature(value, sig_diff_salt) == False
    
    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value_none = b"test-none"
    sig_none = signer_none.get_signature(value_none)
    assert sig_none == b""
    assert signer_none.verify_signature(value_none, sig_none) == True
    
    # Test with custom digest method
    import hashlib
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value_sha256 = b"test-sha256"
    sig_sha256 = signer_sha256.get_signature(value_sha256)
    assert signer_sha256.verify_signature(value_sha256, sig_sha256) == True
    
    # Test with different key derivation
    signer_concat = Signer("secret-key", key_derivation="concat")
    value_concat = b"test-concat"
    sig_concat = signer_concat.get_signature(value_concat)
    assert signer_concat.verify_signature(value_concat, sig_concat) == True
    
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    value_hmac = b"test-hmac"
    sig_hmac = signer_hmac.get_signature(value_hmac)
    assert signer_hmac.verify_signature(value_hmac, sig_hmac) == True
    
    signer_none_deriv = Signer("secret-key", key_derivation="none")
    value_none_deriv = b"test-none-deriv"
    sig_none_deriv = signer_none_deriv.get_signature(value_none_deriv)
    assert signer_none_deriv.verify_signature(value_none_deriv, sig_none_deriv) == True
    
    # Test empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) == True
    
    # Test with special characters in value
    special_value = b"test with spaces and !@#$%^&*()"
    sig_special = signer.get_signature(special_value)
    assert signer.verify_signature(special_value, sig_special) == True
```


# LLM-generated content at query #13
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) is True

    # Test with string value (not bytes)
    assert signer.verify_signature("test-value", sig) is True

    # Test with string signature (not bytes)
    sig_str = sig.decode("ascii")
    assert signer.verify_signature(value, sig_str) is True

    # Test with NoneAlgorithm (no signature)
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) is True
    # Empty signature should still verify with NoneAlgorithm
    assert none_signer.verify_signature(value, b"") is True

    # Test with key rotation - verify with old key
    new_signer = Signer(["old-key", "new-key"], salt="test-salt")
    old_sig = Signer("old-key", salt="test-salt").get_signature(value)
    assert new_signer.verify_signature(value, old_sig) is True

    # Test with key rotation - verify with new key
    new_sig = new_signer.get_signature(value)
    assert new_signer.verify_signature(value, new_sig) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False

    # Test with different salt
    signer2 = Signer("secret-key", salt="different-salt")
    sig2 = signer2.get_signature(value)
    assert signer.verify_signature(value, sig2) is False

    # Test with different key
    signer3 = Signer("different-key", salt="test-salt")
    sig3 = signer3.get_signature(value)
    assert signer.verify_signature(value, sig3) is False

    # Test with HMAC algorithm directly
    hmac_algorithm = HMACAlgorithm()
    hmac_signer = Signer("secret-key", algorithm=hmac_algorithm)
    hmac_sig = hmac_signer.get_signature(value)
    assert hmac_signer.verify_signature(value, hmac_sig) is True
```


# LLM-generated content at query #14
#--------------------------

```python
def test_Signer_verify_signature():
    """Test verify_signature method with various scenarios."""
    signer = Signer("secret-key")
    
    # Test valid signature
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True
    
    # Test invalid signature
    invalid_sig = base64_encode(b"invalid")
    assert signer.verify_signature(value, invalid_sig) == False
    
    # Test with string value
    value_str = "test-string"
    sig_str = signer.get_signature(value_str)
    assert signer.verify_signature(value_str, sig_str) == True
    
    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) == True
    
    # Test with modified value
    modified_value = b"modified-value"
    assert signer.verify_signature(modified_value, sig) == False
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False
    
    # Test with different salt
    signer2 = Signer("secret-key", salt="different-salt")
    sig2 = signer2.get_signature(value)
    assert signer.verify_signature(value, sig2) == False
    assert signer2.verify_signature(value, sig2) == True
    
    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test"
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) == True
    
    # Test with key rotation (multiple secret keys)
    rotated_signer = Signer(["old-key", "new-key"])
    # Sign with newest key
    sig_rotated = rotated_signer.get_signature(value)
    assert rotated_signer.verify_signature(value, sig_rotated) == True
    
    # Test with custom digest method
    custom_signer = Signer("secret-key", digest_method=hashlib.sha256)
    custom_sig = custom_signer.get_signature(value)
    assert custom_signer.verify_signature(value, custom_sig) == True
    assert signer.verify_signature(value, custom_sig) == False
    
    # Test with custom key derivation
    concat_signer = Signer("secret-key", key_derivation="concat")
    concat_sig = concat_signer.get_signature(value)
    assert concat_signer.verify_signature(value, concat_sig) == True
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") == False
    
    # Test with very long value
    long_value = b"a" * 10000
    long_sig = signer.get_signature(long_value)
    assert signer.verify_signature(long_value, long_sig) == True
```


# LLM-generated content at query #15
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid-signature")
    assert signer.verify_signature(value, invalid_sig) == False

    # Test with empty value
    empty_value = b""
    empty_sig = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, empty_sig) == True

    # Test with different secret key (should fail)
    signer2 = Signer("different-secret-key")
    assert signer2.verify_signature(value, sig) == False

    # Test with key rotation - should verify with any key in the list
    signer_rotation = Signer(["old-key", "new-key"])
    old_sig = signer_rotation.get_signature(value)
    # Change secret keys to simulate rotation
    signer_rotation.secret_keys = ["old-key", "new-key"]
    assert signer_rotation.verify_signature(value, old_sig) == True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"not-base64!!") == False

    # Test with NoneAlgorithm
    none_algorithm_signer = Signer("key", algorithm=NoneAlgorithm())
    none_sig = none_algorithm_signer.get_signature(value)
    assert none_algorithm_signer.verify_signature(value, none_sig) == True
    assert none_algorithm_signer.verify_signature(value, b"anything") == True

    # Test with string input instead of bytes
    str_value = "string-value"
    str_sig = signer.get_signature(str_value)
    assert signer.verify_signature(str_value, str_sig) == True

    # Test with HMACAlgorithm and different digest methods
    import hashlib
    sha256_signer = Signer("key", digest_method=hashlib.sha256)
    sha256_sig = sha256_signer.get_signature(value)
    assert sha256_signer.verify_signature(value, sha256_sig) == True
```


# LLM-generated content at query #16
#--------------------------

```python
def test_Signer_verify_signature():
    signer = Signer(secret_key="test-secret-key")
    
    # Test with valid signature
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True
    
    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid")
    assert signer.verify_signature(value, invalid_sig) == False
    
    # Test with empty value
    empty_value = b""
    empty_sig = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, empty_sig) == True
    
    # Test with non-bytes value
    str_value = "test-string"
    str_sig = signer.get_signature(str_value)
    assert signer.verify_signature(str_value, str_sig) == True
    
    # Test with non-bytes signature
    str_sig_str = str_sig.decode("utf-8")
    assert signer.verify_signature(str_value, str_sig_str) == True
    
    # Test with corrupted signature
    corrupted_sig = sig[:-1] + bytes([sig[-1] ^ 0xFF])
    assert signer.verify_signature(value, corrupted_sig) == False
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, "!!!invalid-base64!!!") == False
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") == False
    
    # Test with NoneAlgorithm
    none_signer = Signer(secret_key="test", algorithm=NoneAlgorithm())
    none_value = b"test-value"
    none_sig = none_signer.get_signature(none_value)
    assert none_signer.verify_signature(none_value, none_sig) == True
    
    # Test with key rotation
    rotated_signer = Signer(secret_key=["old-key", "new-key"])
    old_sig = Signer(secret_key="old-key").get_signature(value)
    assert rotated_signer.verify_signature(value, old_sig) == True
    new_sig = rotated_signer.get_signature(value)
    assert rotated_signer.verify_signature(value, new_sig) == True
    
    # Test with invalid key (should not verify)
    different_signer = Signer(secret_key="different-key")
    different_sig = different_signer.get_signature(value)
    assert signer.verify_signature(value, different_sig) == False
```


# LLM-generated content at query #17
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    invalid_signature = base64_encode(b"invalid-sig")
    assert signer.verify_signature(value, invalid_signature) is False

    # Test with empty value
    empty_value = b""
    empty_sig = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, empty_sig) is True

    # Test with different secret key (should fail)
    different_signer = Signer("different-secret", salt="test-salt")
    different_sig = different_signer.get_signature(value)
    assert signer.verify_signature(value, different_sig) is False

    # Test with key rotation: verify with older key
    signer_rotation = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    # Sign with the newest key
    signature = signer_rotation.get_signature(value)
    # Should verify with the newer key
    assert signer_rotation.verify_signature(value, signature) is True

    # Test with corrupted signature (invalid base64)
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False

    # Test with different salt (should fail)
    different_salt_signer = Signer("secret-key", salt="different-salt")
    different_salt_sig = different_salt_signer.get_signature(value)
    assert signer.verify_signature(value, different_salt_sig) is False

    # Test with NoneAlgorithm
    none_alg_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    none_sig = none_alg_signer.get_signature(value)
    assert none_alg_signer.verify_signature(value, none_sig) is True
    # With NoneAlgorithm, any signature should verify
    assert none_alg_signer.verify_signature(value, b"anything") is True
```


# LLM-generated content at query #18
#--------------------------

```python
def test_Signer_verify_signature():
    """Test verify_signature method of Signer class."""
    # Test with default configuration
    signer = Signer("secret-key")
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with string value
    sig_str = signer.get_signature("test_string")
    assert signer.verify_signature("test_string", sig_str) is True
    
    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid_sig") is False
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") is False
    
    # Test with non-base64 signature
    assert signer.verify_signature(value, b"!!!invalid_base64!!!") is False
    
    # Test with different key derivation
    signer_concat = Signer("secret-key", key_derivation="concat")
    sig_concat = signer_concat.get_signature(value)
    assert signer_concat.verify_signature(value, sig_concat) is True
    
    # Test with hmac key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac) is True
    
    # Test with none key derivation
    signer_none = Signer("secret-key", key_derivation="none")
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True
    
    # Test with custom salt
    signer_custom_salt = Signer("secret-key", salt=b"custom_salt")
    sig_custom = signer_custom_salt.get_signature(value)
    assert signer_custom_salt.verify_signature(value, sig_custom) is True
    
    # Test with multiple secret keys (key rotation)
    signer_rotation = Signer(["old-key", "new-key"])
    sig_rotation = signer_rotation.get_signature(value)
    assert signer_rotation.verify_signature(value, sig_rotation) is True
    
    # Test that signature from old key still verifies
    old_signer = Signer("old-key")
    old_sig = old_signer.get_signature(value)
    assert signer_rotation.verify_signature(value, old_sig) is True
    
    # Test with NoneAlgorithm
    signer_none_alg = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none_alg = signer_none_alg.get_signature(value)
    assert signer_none_alg.verify_signature(value, sig_none_alg) is True
    assert sig_none_alg == b""
    
    # Test with HMACAlgorithm and custom digest
    import hashlib
    signer_sha256 = Signer("secret-key", algorithm=HMACAlgorithm(hashlib.sha256))
    sig_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig_sha256) is True
    
    # Test that different keys produce different signatures
    signer1 = Signer("key1")
    signer2 = Signer("key2")
    sig1 = signer1.get_signature(value)
    sig2 = signer2.get_signature(value)
    assert signer1.verify_signature(value, sig1) is True
    assert signer1.verify_signature(value, sig2) is False
    assert signer2.verify_signature(value, sig2) is True
    assert signer2.verify_signature(value, sig1) is False
    
    # Test with unicode/bytes mixed inputs
    unicode_value = "héllo"
    sig_unicode = signer.get_signature(unicode_value)
    assert signer.verify_signature(unicode_value, sig_unicode) is True
    assert signer.verify_signature(unicode_value.encode(), sig_unicode) is True
```


# LLM-generated content at query #19
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with a simple secret key and salt
    signer = Signer(secret_key="my-secret-key", salt="my-salt")
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with wrong signature
    wrong_sig = base64_encode(b"wrong_signature")
    assert signer.verify_signature(value, wrong_sig) is False
    
    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) is True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, "invalid_base64!!") is False
    
    # Test with multiple secret keys (key rotation)
    signer_rotation = Signer(secret_key=["old-key", "new-key"])
    value_rotation = b"rotation_test"
    sig_rotation = signer_rotation.get_signature(value_rotation)
    
    # Verify with newest key
    assert signer_rotation.verify_signature(value_rotation, sig_rotation) is True
    
    # Verify with old key should still work
    old_signer = Signer(secret_key="old-key")
    old_sig = old_signer.get_signature(value_rotation)
    assert signer_rotation.verify_signature(value_rotation, old_sig) is True
    
    # Test with NoneAlgorithm
    signer_none = Signer(secret_key="key", algorithm=NoneAlgorithm())
    value_none = b"test_none"
    sig_none = signer_none.get_signature(value_none)
    assert signer_none.verify_signature(value_none, sig_none) is True


# LLM-generated content at query #20
#--------------------------

```python
def test_Signer_verify_signature():
    """Test that verify_signature returns True for valid signatures and False for invalid ones."""
    signer = Signer("test-secret-key")
    
    # Test valid signature
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True
    
    # Test invalid signature
    invalid_sig = b"invalid-signature"
    assert signer.verify_signature(value, invalid_sig) == False
    
    # Test with different value
    different_value = b"different-value"
    assert signer.verify_signature(different_value, sig) == False
    
    # Test with string inputs
    value_str = "test-string"
    sig_str = signer.get_signature(value_str)
    assert signer.verify_signature(value_str, sig_str) == True
    
    # Test with empty value
    empty_value = b""
    empty_sig = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, empty_sig) == True
    
    # Test with NoneAlgorithm
    none_signer = Signer("test-key", algorithm=NoneAlgorithm())
    value = b"test"
    sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, sig) == True
    
    # Test with malformed signature (not valid base64)
    assert signer.verify_signature(b"test", b"!!!") == False
    
    # Test with key rotation
    signer_rotation = Signer(["old-key", "new-key"])
    value = b"test-rotation"
    sig = signer_rotation.get_signature(value)
    assert signer_rotation.verify_signature(value, sig) == True
    
    # Test with different key derivation
    concat_signer = Signer("test-key", key_derivation="concat")
    value = b"test-concat"
    sig = concat_signer.get_signature(value)
    assert concat_signer.verify_signature(value, sig) == True
    
    hmac_signer = Signer("test-key", key_derivation="hmac")
    value = b"test-hmac"
    sig = hmac_signer.get_signature(value)
    assert hmac_signer.verify_signature(value, sig) == True
    
    none_derivation_signer = Signer("test-key", key_derivation="none")
    value = b"test-none"
    sig = none_derivation_signer.get_signature(value)
    assert none_derivation_signer.verify_signature(value, sig) == True
    
    # Test with custom separator
    custom_sep_signer = Signer("test-key", sep=b"|")
    value = b"test-custom-sep"
    sig = custom_sep_signer.get_signature(value)
    assert custom_sep_signer.verify_signature(value, sig) == True
```


# LLM-generated content at query #21
#--------------------------

```python
def test_Signer_verify_signature():
    """Test verify_signature method of Signer class."""
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False
    
    # Test with empty value
    empty_value = b""
    empty_sig = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, empty_sig) is True
    
    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(sig)) is True
    
    # Test with string value
    str_value = "string-value"
    str_sig = signer.get_signature(str_value)
    assert signer.verify_signature(str_value, str_sig) is True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False
    
    # Test with key rotation - verify with older key
    old_signer = Signer("old-secret-key", salt="test-salt")
    old_value = b"old-value"
    old_sig = old_signer.get_signature(old_value)
    
    new_signer = Signer(["old-secret-key", "new-secret-key"], salt="test-salt")
    assert new_signer.verify_signature(old_value, old_sig) is True
    
    # Test with different algorithm
    none_alg_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    none_sig = none_alg_signer.get_signature(value)
    assert none_alg_signer.verify_signature(value, none_sig) is True
    
    # Test with different key derivation
    hmac_signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    hmac_sig = hmac_signer.get_signature(value)
    assert hmac_signer.verify_signature(value, hmac_sig) is True
    
    # Test with different digest method
    sha256_signer = Signer("secret-key", salt="test-salt", digest_method=hashlib.sha256)
    sha256_sig = sha256_signer.get_signature(value)
    assert sha256_signer.verify_signature(value, sha256_sig) is True
    
    # Test with None salt
    none_salt_signer = Signer("secret-key", salt=None)
    none_salt_sig = none_salt_signer.get_signature(value)
    assert none_salt_signer.verify_signature(value, none_salt_sig) is True
    
    # Test with bytes salt
    bytes_salt_signer = Signer("secret-key", salt=b"bytes-salt")
    bytes_salt_sig = bytes_salt_signer.get_signature(value)
    assert bytes_salt_signer.verify_signature(value, bytes_salt_sig) is True
    
    # Test with empty string value
    empty_str = ""
    empty_str_sig = signer.get_signature(empty_str)
    assert signer.verify_signature(empty_str, empty_str_sig) is True
    
    # Test with NoneAlgorithm returns empty signature
    none_alg = NoneAlgorithm()
    none_alg_signer = Signer("secret-key", salt="test-salt", algorithm=none_alg)
    sig_from_none = none_alg_signer.get_signature(value)
    assert sig_from_none == base64_encode(b"")
    assert none_alg_signer.verify_signature(value, sig_from_none) is True
    
    # Test with HMACAlgorithm directly
    hmac_alg = HMACAlgorithm(hashlib.sha256)
    hmac_alg_signer = Signer("secret-key", salt="test-salt", algorithm=hmac_alg)
    hmac_alg_sig = hmac_alg_signer.get_signature(value)
    assert hmac_alg_signer.verify_signature(value, hmac_alg_sig) is True
```


# LLM-generated content at query #22
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid-signature")
    assert signer.verify_signature(value, invalid_sig) is False

    # Test with empty signature
    assert signer.verify_signature(value, b"") is False

    # Test with bytes value and string sig
    sig_str = sig.decode("ascii")
    assert signer.verify_signature(value, sig_str) is True

    # Test with string value and bytes sig
    value_str = value.decode("utf-8")
    assert signer.verify_signature(value_str, sig) is True

    # Test with key rotation - verify with older key
    signer_with_rotation = Signer(["old-key", "new-key"])
    value2 = b"another-value"
    sig2 = signer_with_rotation.get_signature(value2)
    # Should verify with any key in the list
    assert signer_with_rotation.verify_signature(value2, sig2) is True

    # Test with NoneAlgorithm
    none_algorithm_signer = Signer("key", algorithm=NoneAlgorithm())
    value3 = b"test"
    sig3 = none_algorithm_signer.get_signature(value3)
    assert none_algorithm_signer.verify_signature(value3, sig3) is True

    # Test with modified value
    modified_value = b"modified-value"
    assert signer.verify_signature(modified_value, sig) is False

    # Test with different salt
    signer_different_salt = Signer("secret-key", salt=b"different-salt")
    assert signer_different_salt.verify_signature(value, sig) is False
```


# LLM-generated content at query #23
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid_signature") == False

    # Test with empty value
    assert signer.verify_signature(b"", sig) == False

    # Test with corrupted signature (base64 decode failure)
    assert signer.verify_signature(value, b"!!!not_base64!!!") == False

    # Test with string value
    value_str = "test string"
    sig_str = signer.get_signature(value_str)
    assert signer.verify_signature(value_str, sig_str) == True

    # Test with multiple secret keys (key rotation)
    signer_rotation = Signer(["old-key", "new-key"], salt="salt")
    value_rotation = b"rotation test"
    sig_rotation = signer_rotation.get_signature(value_rotation)
    assert signer_rotation.verify_signature(value_rotation, sig_rotation) == True

    # Test with older key still works
    old_signer = Signer("old-key", salt="salt")
    old_sig = old_signer.get_signature(value_rotation)
    assert signer_rotation.verify_signature(value_rotation, old_sig) == True

    # Test with NoneAlgorithm
    none_algorithm = NoneAlgorithm()
    signer_none = Signer("secret", algorithm=none_algorithm)
    value_none = b"none algo"
    sig_none = signer_none.get_signature(value_none)
    assert signer_none.verify_signature(value_none, sig_none) == True
    assert signer_none.verify_signature(value_none, b"") == True  # NoneAlgorithm returns empty signature

    # Test with HMACAlgorithm and different digest
    hmac_algo = HMACAlgorithm(digest_method=hashlib.sha256)
    signer_hmac = Signer("secret", algorithm=hmac_algo)
    value_hmac = b"hmac test"
    sig_hmac = signer_hmac.get_signature(value_hmac)
    assert signer_hmac.verify_signature(value_hmac, sig_hmac) == True
```


# LLM-generated content at query #24
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid")
    assert signer.verify_signature(value, invalid_sig) is False
    
    # Test with empty value
    value_empty = b""
    sig_empty = signer.get_signature(value_empty)
    assert signer.verify_signature(value_empty, sig_empty) is True
    
    # Test with invalid base64
    assert signer.verify_signature(b"test", b"!!!invalid-base64!!!") is False
    
    # Test with different keys (key rotation)
    signer_multi = Signer(["old-key", "new-key"])
    # Sign with newest key
    value_multi = b"test-multi"
    sig_multi = signer_multi.get_signature(value_multi)
    # Verify with both keys should work
    assert signer_multi.verify_signature(value_multi, sig_multi) is True
    # Verify with only old key should fail (signature made with new key)
    signer_old_only = Signer("old-key")
    # Since get_signature uses the newest key, this will use "new-key"
    assert signer_old_only.verify_signature(value_multi, sig_multi) is False
    
    # Test with string input for value
    assert signer.verify_signature("test-string", sig) is False  # different value
    sig_str = signer.get_signature("test-string")
    assert signer.verify_signature("test-string", sig_str) is True
    
    # Test with NoneAlgorithm
    none_signer = Signer("key", algorithm=NoneAlgorithm())
    value_none = b"test-none"
    sig_none = none_signer.get_signature(value_none)
    assert sig_none == base64_encode(b"")  # empty signature
    assert none_signer.verify_signature(value_none, sig_none) is True
    # Empty base64 decoded is b"", which matches NoneAlgorithm's empty signature
    assert none_signer.verify_signature(value_none, base64_encode(b"")) is True
    # Any non-empty signature should fail
    assert none_signer.verify_signature(value_none, base64_encode(b"x")) is False
    
    # Test with different digest methods
    signer_sha256 = Signer("key", digest_method=hashlib.sha256)
    value_sha = b"test-sha256"
    sig_sha = signer_sha256.get_signature(value_sha)
    assert signer_sha256.verify_signature(value_sha, sig_sha) is True
    # Should fail with default signer
    default_signer = Signer("key")
    assert default_signer.verify_signature(value_sha, sig_sha) is False
```


# LLM-generated content at query #25
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with invalid signature
    invalid_sig = b"invalid-signature"
    assert signer.verify_signature(value, invalid_sig) is False
    
    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) is True
    
    # Test with different secret keys (key rotation)
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"rotation-test"
    sig_new = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_new) is True
    
    # Test with binary value
    binary_value = bytes(range(256))
    sig_binary = signer.get_signature(binary_value)
    assert signer.verify_signature(binary_value, sig_binary) is True
    
    # Test with string value (converted to bytes)
    string_value = "test-string"
    sig_string = signer.get_signature(string_value)
    assert signer.verify_signature(string_value, sig_string) is True
    
    # Test with NoneAlgorithm
    none_signer = Signer("secret", algorithm=NoneAlgorithm())
    value = b"test-none"
    sig_none = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, sig_none) is True
    
    # Test with different key derivation methods
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_derivation = Signer("secret", key_derivation=derivation)
        value = b"derivation-test"
        sig_derivation = signer_derivation.get_signature(value)
        assert signer_derivation.verify_signature(value, sig_derivation) is True
    
    # Test with invalid base64 signature (should return False)
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") is False
    
    # Test with NoneAlgorithm with empty signature
    none_signer = Signer("secret", algorithm=NoneAlgorithm())
    assert none_signer.verify_signature(b"test", b"") is True
```


# LLM-generated content at query #26
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    
    # Verify correct signature returns True
    assert signer.verify_signature(value, sig) == True
    
    # Verify incorrect signature returns False
    assert signer.verify_signature(value, b"invalid") == False
    
    # Verify with string value
    assert signer.verify_signature("test value", sig) == True
    
    # Test with modified value
    assert signer.verify_signature(b"modified value", sig) == False
    
    # Test with empty value
    empty_sig = signer.get_signature(b"")
    assert signer.verify_signature(b"", empty_sig) == True
    
    # Test with key rotation (list of keys)
    signer_rotation = Signer(["old-key", "new-key"])
    value_rotation = b"rotation test"
    sig_rotation = signer_rotation.get_signature(value_rotation)
    assert signer_rotation.verify_signature(value_rotation, sig_rotation) == True
    
    # Test that old key still works for verification
    signer_old_key = Signer("old-key")
    sig_old = signer_old_key.get_signature(value_rotation)
    assert signer_rotation.verify_signature(value_rotation, sig_old) == True
    
    # Test with different algorithm (NoneAlgorithm)
    none_signer = Signer("secret", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(b"test")
    # With NoneAlgorithm, any signature should verify (since get_signature returns empty bytes)
    assert none_signer.verify_signature(b"test", none_sig) == True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(b"test", b"!!!invalid base64!!!") == False
    
    # Test with different salt
    signer_salt = Signer("secret", salt="custom-salt")
    sig_salt = signer_salt.get_signature(b"test")
    assert signer_salt.verify_signature(b"test", sig_salt) == True
    
    # Test with different separator
    signer_sep = Signer("secret", sep=b"|")
    sig_sep = signer_sep.get_signature(b"test")
    assert signer_sep.verify_signature(b"test", sig_sep) == True
    
    # Test with different digest method (SHA256)
    import hashlib
    signer_sha256 = Signer("secret", digest_method=hashlib.sha256)
    sig_sha256 = signer_sha256.get_signature(b"test")
    assert signer_sha256.verify_signature(b"test", sig_sha256) == True
    
    # Test with different key derivation
    signer_concat = Signer("secret", key_derivation="concat")
    sig_concat = signer_concat.get_signature(b"test")
    assert signer_concat.verify_signature(b"test", sig_concat) == True
    
    signer_hmac = Signer("secret", key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(b"test")
    assert signer_hmac.verify_signature(b"test", sig_hmac) == True
    
    signer_none = Signer("secret", key_derivation="none")
    sig_none = signer_none.get_signature(b"test")
    assert signer_none.verify_signature(b"test", sig_none) == True
    
    # Test with bytes and string secret keys
    signer_bytes = Signer(b"bytes-key")
    sig_bytes = signer_bytes.get_signature(b"test")
    assert signer_bytes.verify_signature(b"test", sig_bytes) == True
    
    # Test that NoneAlgorithm with any signature returns True
    none_alg = NoneAlgorithm()
    assert none_alg.verify_signature(b"key", b"value", b"") == True
    assert none_alg.verify_signature(b"key", b"value", b"anything") == True
```


# LLM-generated content at query #27
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

    # Test with string value
    value_str = "test-string"
    sig_str = signer.get_signature(value_str)
    assert signer.verify_signature(value_str, sig_str) == True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") == False

    # Test with empty signature
    assert signer.verify_signature(value, b"") == False

    # Test with NoneAlgorithm
    none_signer = Signer("secret", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) == True
    assert none_signer.verify_signature(value, b"") == True

    # Test with base64 invalid input
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False

    # Test with key rotation
    signer_rotation = Signer(["old-key", "new-key"])
    value_rot = b"rotate-test"
    sig_rot = signer_rotation.get_signature(value_rot)
    assert signer_rotation.verify_signature(value_rot, sig_rot) == True

    # Test with different key derivation methods
    hmac_signer = Signer("secret", key_derivation="hmac")
    sig_hmac = hmac_signer.get_signature(value)
    assert hmac_signer.verify_signature(value, sig_hmac) == True

    concat_signer = Signer("secret", key_derivation="concat")
    sig_concat = concat_signer.get_signature(value)
    assert concat_signer.verify_signature(value, sig_concat) == True

    # Test with custom digest method
    sha256_signer = Signer("secret", digest_method=hashlib.sha256)
    sig_sha256 = sha256_signer.get_signature(value)
    assert sha256_signer.verify_signature(value, sig_sha256) == True

    # Test with bytes separator
    signer_sep = Signer("secret", sep=b"|")
    value_sep = b"test-sep"
    sig_sep = signer_sep.get_signature(value_sep)
    assert signer_sep.verify_signature(value_sep, sig_sep) == True
```


# LLM-generated content at query #28
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with string value
    value_str = "test string"
    sig_str = signer.get_signature(value_str)
    assert signer.verify_signature(value_str, sig_str) is True
    
    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid") is False
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") is False
    
    # Test with key rotation (list of keys)
    signer_rotation = Signer(["old-key", "new-key"])
    value_rot = b"rotated value"
    sig_rot = signer_rotation.get_signature(value_rot)
    # Should verify with the newest key
    assert signer_rotation.verify_signature(value_rot, sig_rot) is True
    
    # Test with NoneAlgorithm
    none_signer = Signer("secret", algorithm=NoneAlgorithm())
    value_none = b"none test"
    sig_none = none_signer.get_signature(value_none)
    assert none_signer.verify_signature(value_none, sig_none) is True
    
    # Test with custom digest method
    custom_signer = Signer("secret", digest_method=hashlib.sha256)
    value_custom = b"custom test"
    sig_custom = custom_signer.get_signature(value_custom)
    assert custom_signer.verify_signature(value_custom, sig_custom) is True
    
    # Test signature from different key fails
    other_signer = Signer("other-key")
    value_diff = b"different key"
    sig_diff = other_signer.get_signature(value_diff)
    assert signer.verify_signature(value_diff, sig_diff) is False
    
    # Test with key derivation methods
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_deriv = Signer("secret", key_derivation=derivation)
        value_deriv = b"derivation test"
        sig_deriv = signer_deriv.get_signature(value_deriv)
        assert signer_deriv.verify_signature(value_deriv, sig_deriv) is True
    
    # Test with all secret keys in rotation verify the same value
    old_signer = Signer("old-key")
    value_old = b"old value"
    sig_old = old_signer.get_signature(value_old)
    
    new_signer = Signer(["old-key", "new-key"])
    assert new_signer.verify_signature(value_old, sig_old) is True
    
    # Test with malformed base64 signature
    assert signer.verify_signature(b"test", b"!!!invalid base64!!!") is False
    
    # Test with binary value containing separator
    binary_value = b"test" + b"." + b"value"
    sig_binary = signer.get_signature(binary_value)
    assert signer.verify_signature(binary_value, sig_binary) is True
    
    # Test with different salt
    signer_salt1 = Signer("secret", salt="salt1")
    signer_salt2 = Signer("secret", salt="salt2")
    value_salt = b"salt test"
    sig_salt1 = signer_salt1.get_signature(value_salt)
    assert signer_salt1.verify_signature(value_salt, sig_salt1) is True
    assert signer_salt2.verify_signature(value_salt, sig_salt1) is False
```


# LLM-generated content at query #29
#--------------------------

```python
def test_Signer_verify_signature():
    """Test verify_signature method of Signer class."""
    signer = Signer(secret_key="test-secret-key", salt="test-salt")
    
    # Test with valid signature
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True
    
    # Test with invalid signature
    invalid_sig = b"invalid-signature"
    assert signer.verify_signature(value, invalid_sig) == False
    
    # Test with empty value
    empty_value = b""
    empty_sig = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, empty_sig) == True
    
    # Test with string input for value
    assert signer.verify_signature("test-value", sig) == True
    
    # Test with string input for sig
    assert signer.verify_signature(value, sig.decode()) == True
    
    # Test with modified value
    modified_value = b"modified-value"
    assert signer.verify_signature(modified_value, sig) == False
    
    # Test with key rotation
    signer_rotated = Signer(
        secret_key=["old-key", "new-key"],
        salt="test-salt"
    )
    # Sign with newest key (last in list)
    rotated_sig = signer_rotated.get_signature(value)
    # Verify with all keys (should work with newest)
    assert signer_rotated.verify_signature(value, rotated_sig) == True
    
    # Test with NoneAlgorithm
    none_algorithm = NoneAlgorithm()
    signer_none = Signer(
        secret_key="test-key",
        algorithm=none_algorithm
    )
    none_sig = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, none_sig) == True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False
    
    # Test with bytes that decode to invalid signature
    invalid_decoded = base64_encode(b"\xff\xff\xff")
    assert signer.verify_signature(value, invalid_decoded) == False
    
    # Test with different digest methods
    import hashlib
    signer_sha256 = Signer(
        secret_key="test-key",
        digest_method=hashlib.sha256
    )
    sha256_sig = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sha256_sig) == True
    assert signer_sha256.verify_signature(value, sig) == False
```


# LLM-generated content at query #30
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default configuration
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    
    # Valid signature should return True
    assert signer.verify_signature(value, sig) is True
    
    # Invalid signature should return False
    assert signer.verify_signature(value, b"invalid-signature") is False
    
    # Invalid base64 signature should return False
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False
    
    # Test with string value
    assert signer.verify_signature("test-value", sig) is True
    
    # Test with key rotation
    signer_rotation = Signer(["old-key", "new-key"])
    value2 = b"test-value-2"
    sig2 = signer_rotation.get_signature(value2)
    
    # Should verify with newest key
    assert signer_rotation.verify_signature(value2, sig2) is True
    
    # Should verify with old key if signed with it
    old_signer = Signer("old-key")
    old_sig = old_signer.get_signature(value2)
    assert signer_rotation.verify_signature(value2, old_sig) is True
    
    # Should not verify with completely wrong key
    wrong_signer = Signer("wrong-key")
    wrong_sig = wrong_signer.get_signature(value2)
    assert signer_rotation.verify_signature(value2, wrong_sig) is False
    
    # Test with different salt
    signer_salt = Signer("secret-key", salt="custom-salt")
    value3 = b"test-value-3"
    sig3 = signer_salt.get_signature(value3)
    assert signer_salt.verify_signature(value3, sig3) is True
    
    # Signature from different salt should not verify
    default_signer = Signer("secret-key")
    default_sig = default_signer.get_signature(value3)
    assert signer_salt.verify_signature(value3, default_sig) is False
    
    # Test with different separator
    signer_sep = Signer("secret-key", sep=b"|")
    value4 = b"test-value-4"
    sig4 = signer_sep.get_signature(value4)
    assert signer_sep.verify_signature(value4, sig4) is True
    
    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value5 = b"test-value-5"
    sig5 = none_signer.get_signature(value5)
    assert none_signer.verify_signature(value5, sig5) is True
```


# LLM-generated content at query #31
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    
    # Test with valid signature
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True
    
    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid") == False
    
    # Test with empty value
    assert signer.verify_signature(b"", sig) == False
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!") == False
    
    # Test with non-bytes value
    assert signer.verify_signature("test value", sig) == True
    
    # Test with key rotation (multiple keys)
    signer_rotation = Signer(["old-key", "new-key"])
    value = b"test with rotation"
    sig = signer_rotation.get_signature(value)
    assert signer_rotation.verify_signature(value, sig) == True
    
    # Test with different salt
    signer_diff_salt = Signer("secret-key", salt=b"different-salt")
    assert signer_diff_salt.verify_signature(value, sig) == False
    
    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test none algorithm"
    sig = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig) == True
    
    # Test with HMACAlgorithm and custom digest
    signer_hmac = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test hmac sha256"
    sig = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig) == True
    assert signer_hmac.verify_signature(value, b"wrong") == False
```


# LLM-generated content at query #32
#--------------------------

```python
def test_Signer_verify_signature():
    """Test verify_signature method of Signer class."""
    # Test with valid signature
    signer = Signer(secret_key="secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid-signature")
    assert signer.verify_signature(value, invalid_sig) is False

    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) is True

    # Test with NoneAlgorithm
    none_alg_signer = Signer(secret_key="secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    sig_none = none_alg_signer.get_signature(value)
    assert none_alg_signer.verify_signature(value, sig_none) is True

    # Test with HMACAlgorithm and custom digest method
    hmac_signer = Signer(secret_key="secret-key", salt="test-salt", digest_method=hashlib.sha256)
    sig_hmac = hmac_signer.get_signature(value)
    assert hmac_signer.verify_signature(value, sig_hmac) is True

    # Test with key rotation
    key_rotation_signer = Signer(secret_key=["old-key", "new-key"], salt="test-salt")
    sig_key_rot = key_rotation_signer.get_signature(value)
    assert key_rotation_signer.verify_signature(value, sig_key_rot) is True

    # Test with corrupted signature (base64 decode failure)
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False

    # Test with string value
    string_value = "test-string"
    sig_string = signer.get_signature(string_value)
    assert signer.verify_signature(string_value, sig_string) is True

    # Test with different salt
    signer2 = Signer(secret_key="secret-key", salt="different-salt")
    sig_diff_salt = signer2.get_signature(value)
    assert signer.verify_signature(value, sig_diff_salt) is False
    assert signer2.verify_signature(value, sig_diff_salt) is True

    # Test with different separator
    sep_signer = Signer(secret_key="secret-key", salt="test-salt", sep=b"|")
    sig_sep = sep_signer.get_signature(value)
    assert sep_signer.verify_signature(value, sig_sep) is True

    # Test verify_signature returns False for empty signature
    assert signer.verify_signature(value, b"") is False

    # Test with bytes value that contains separator
    value_with_sep = b"test.value"
    sig_with_sep = signer.get_signature(value_with_sep)
    assert signer.verify_signature(value_with_sep, sig_with_sep) is True
```


# LLM-generated content at query #33
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default Signer configuration
    signer = Signer(secret_key="test-secret-key")
    
    # Test basic verification
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with string input
    assert signer.verify_signature("test-value", sig) is True
    
    # Test with wrong signature
    wrong_sig = base64_encode(b"wrong-signature")
    assert signer.verify_signature(value, wrong_sig) is False
    
    # Test with corrupted base64 signature
    assert signer.verify_signature(value, "!!!invalid-base64!!!") is False
    
    # Test with empty string
    sig_empty = signer.get_signature(b"")
    assert signer.verify_signature(b"", sig_empty) is True
    
    # Test with key rotation (multiple secret keys)
    rotated_signer = Signer(secret_key=["old-key", "new-key"])
    value2 = b"rotated-value"
    sig2 = rotated_signer.get_signature(value2)
    assert rotated_signer.verify_signature(value2, sig2) is True
    
    # Test with old key still works (key rotation support)
    old_signer = Signer(secret_key="old-key")
    sig_old = old_signer.get_signature(value2)
    assert rotated_signer.verify_signature(value2, sig_old) is True
    
    # Test with NoneAlgorithm
    none_signer = Signer(secret_key="test", algorithm=NoneAlgorithm())
    value3 = b"unsigned-value"
    sig3 = none_signer.get_signature(value3)
    assert none_signer.verify_signature(value3, sig3) is True
    
    # Test with HMACAlgorithm and custom digest
    hmac_signer = Signer(
        secret_key="test",
        algorithm=HMACAlgorithm(digest_method=hashlib.sha256)
    )
    value4 = b"hmac-value"
    sig4 = hmac_signer.get_signature(value4)
    assert hmac_signer.verify_signature(value4, sig4) is True
    
    # Test with different salt
    signer_salt1 = Signer(secret_key="test", salt=b"salt1")
    signer_salt2 = Signer(secret_key="test", salt=b"salt2")
    value5 = b"different-salt"
    sig_salt1 = signer_salt1.get_signature(value5)
    assert signer_salt1.verify_signature(value5, sig_salt1) is True
    assert signer_salt2.verify_signature(value5, sig_salt1) is False
    
    # Test with different key derivation methods
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_deriv = Signer(
            secret_key="test",
            key_derivation=derivation
        )
        value6 = f"derivation-{derivation}".encode()
        sig6 = signer_deriv.get_signature(value6)
        assert signer_deriv.verify_signature(value6, sig6) is True
    
    # Test with bytes signature
    value7 = b"bytes-test"
    sig7 = signer.get_signature(value7)
    assert signer.verify_signature(value7, sig7) is True
    
    # Test with signature as bytes
    assert signer.verify_signature(value7, bytes(sig7)) is True
```


# LLM-generated content at query #34
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer(secret_key="secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid-signature")
    assert signer.verify_signature(value, invalid_sig) is False

    # Test with empty signature
    assert signer.verify_signature(value, b"") is False

    # Test with NoneAlgorithm
    signer_none = Signer(secret_key="secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig) is True

    # Test with key rotation (multiple keys)
    signer_rotation = Signer(
        secret_key=["old-key", "new-key"],
        salt="test-salt"
    )
    value = b"test-value"
    sig = signer_rotation.get_signature(value)
    assert signer_rotation.verify_signature(value, sig) is True

    # Test that old keys still work for verification
    old_signer = Signer(secret_key="old-key", salt="test-salt")
    old_sig = old_signer.get_signature(value)
    assert signer_rotation.verify_signature(value, old_sig) is True

    # Test with string value (not bytes)
    assert signer.verify_signature("test-value", sig) is True

    # Test with string signature
    assert signer.verify_signature(value, sig.decode()) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, "not-base64!!") is False

    # Test with different separator
    signer_custom_sep = Signer(secret_key="secret-key", salt="test-salt", sep=b"-")
    value = b"test-value"
    sig = signer_custom_sep.get_signature(value)
    assert signer_custom_sep.verify_signature(value, sig) is True

    # Test with empty value
    assert signer.verify_signature(b"", sig) is False  # signature was for different value

    # Test with different salt
    signer_diff_salt = Signer(secret_key="secret-key", salt="different-salt")
    assert signer_diff_salt.verify_signature(value, sig) is False  # different salt should fail

    # Test with different key derivation
    signer_concat = Signer(
        secret_key="secret-key",
        salt="test-salt",
        key_derivation="concat"
    )
    value = b"test-value"
    sig_concat = signer_concat.get_signature(value)
    assert signer_concat.verify_signature(value, sig_concat) is True
    assert signer.verify_signature(value, sig_concat) is False  # different derivation should fail
```


# LLM-generated content at query #35
#--------------------------

```python
def test_Signer_verify_signature():
    """Test verify_signature method with various scenarios."""
    # Test with valid signature
    signer = Signer("test-secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False
    
    # Test with modified value
    invalid_sig = signer.get_signature(b"different-value")
    assert signer.verify_signature(value, invalid_sig) is False
    
    # Test with bytes value and string sig
    sig_str = sig.decode("ascii")
    assert signer.verify_signature(value, sig_str) is True
    
    # Test with string value and bytes sig
    value_str = value.decode("utf-8")
    assert signer.verify_signature(value_str, sig) is True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False
    
    # Test with empty value
    empty_sig = signer.get_signature(b"")
    assert signer.verify_signature(b"", empty_sig) is True
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") is False
    
    # Test with key rotation - use oldest key for signing
    signer_with_rotation = Signer(
        ["old-key", "new-key"],
        salt="test-salt"
    )
    value = b"test-value"
    sig = signer_with_rotation.get_signature(value)
    # Should verify with any key from the list
    assert signer_with_rotation.verify_signature(value, sig) is True
    
    # Test with NoneAlgorithm
    none_signer = Signer(
        "test-secret-key",
        algorithm=NoneAlgorithm()
    )
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) is True
    
    # Test with different separator
    signer_custom_sep = Signer("test-secret-key", sep=b"|")
    value = b"test-value"
    sig = signer_custom_sep.get_signature(value)
    assert signer_custom_sep.verify_signature(value, sig) is True
    
    # Test with HMAC algorithm
    hmac_signer = Signer(
        "test-secret-key",
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    value = b"test-value"
    sig = hmac_signer.get_signature(value)
    assert hmac_signer.verify_signature(value, sig) is True
    
    # Test that different salt produces different signature
    signer1 = Signer("test-secret-key", salt="salt1")
    signer2 = Signer("test-secret-key", salt="salt2")
    sig1 = signer1.get_signature(value)
    sig2 = signer2.get_signature(value)
    assert sig1 != sig2
    # Signature from signer1 should not verify with signer2
    assert signer2.verify_signature(value, sig1) is False
    
    # Test with unicode value
    unicode_value = "héllo wörld"
    sig = signer.get_signature(unicode_value)
    assert signer.verify_signature(unicode_value, sig) is True
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key()
    expected = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert key == expected

    # Test with concat key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    key = signer.derive_key()
    expected = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert key == expected

    # Test with hmac key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected = mac.digest()
    assert key == expected

    # Test with none key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret-key"

    # Test with custom secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key("custom-key")
    expected = hashlib.sha1(b"test-salt" + b"signer" + b"custom-key").digest()
    assert key == expected

    # Test with bytes secret_key
    signer = Signer(b"secret-key", salt=b"test-salt")
    key = signer.derive_key()
    expected = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert key == expected

    # Test that derive_key raises TypeError for unknown derivation method
    signer = Signer("secret-key", salt="test-salt", key_derivation="unknown")
    try:
        signer.derive_key()
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Unknown key derivation method"
```


# LLM-generated content at query #2
#--------------------------

```python
def test_Signer_derive_key():
    # Test default key derivation (django-concat)
    signer = Signer("test-secret-key", salt="test-salt")
    derived = signer.derive_key()
    expected = hashlib.sha1(b"test-salt" + b"signer" + b"test-secret-key").digest()
    assert derived == expected

    # Test concat derivation
    signer_concat = Signer("test-secret-key", salt="test-salt", key_derivation="concat")
    derived_concat = signer_concat.derive_key()
    expected_concat = hashlib.sha1(b"test-salt" + b"test-secret-key").digest()
    assert derived_concat == expected_concat

    # Test hmac derivation
    signer_hmac = Signer("test-secret-key", salt="test-salt", key_derivation="hmac")
    derived_hmac = signer_hmac.derive_key()
    mac = hmac.new(b"test-secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_hmac = mac.digest()
    assert derived_hmac == expected_hmac

    # Test none derivation
    signer_none = Signer("test-secret-key", salt="test-salt", key_derivation="none")
    derived_none = signer_none.derive_key()
    expected_none = b"test-secret-key"
    assert derived_none == expected_none

    # Test with explicit secret_key parameter
    derived_explicit = signer.derive_key(secret_key="explicit-key")
    expected_explicit = hashlib.sha1(b"test-salt" + b"signer" + b"explicit-key").digest()
    assert derived_explicit == expected_explicit

    # Test with bytes secret_key
    signer_bytes = Signer(b"bytes-secret", salt=b"bytes-salt")
    derived_bytes = signer_bytes.derive_key()
    expected_bytes = hashlib.sha1(b"bytes-salt" + b"signer" + b"bytes-secret").digest()
    assert derived_bytes == expected_bytes

    # Test with custom digest method
    signer_md5 = Signer("test-secret", salt="test-salt", digest_method=hashlib.md5)
    derived_md5 = signer_md5.derive_key()
    expected_md5 = hashlib.md5(b"test-salt" + b"signer" + b"test-secret").digest()
    assert derived_md5 == expected_md5

    # Test with key rotation - uses last key by default
    signer_rotation = Signer(["old-key", "new-key"], salt="test-salt")
    derived_rotation = signer_rotation.derive_key()
    expected_rotation = hashlib.sha1(b"test-salt" + b"signer" + b"new-key").digest()
    assert derived_rotation == expected_rotation

    # Test with key rotation and explicit secret_key
    derived_rotation_explicit = signer_rotation.derive_key(secret_key="old-key")
    expected_rotation_explicit = hashlib.sha1(b"test-salt" + b"signer" + b"old-key").digest()
    assert derived_rotation_explicit == expected_rotation_explicit

    # Test invalid key derivation raises TypeError
    signer_invalid = Signer("test-key", salt="test-salt", key_derivation="invalid")
    try:
        signer_invalid.derive_key()
        assert False, "Expected TypeError for invalid key derivation"
    except TypeError:
        pass

    # Test with no salt
    signer_no_salt = Signer("test-key", salt=None)
    derived_no_salt = signer_no_salt.derive_key()
    expected_no_salt = hashlib.sha1(b"itsdangerous.Signer" + b"signer" + b"test-key").digest()
    assert derived_no_salt == expected_no_salt

    # Test with string salt parameter conversion to bytes
    signer_str_salt = Signer("test-key", salt="custom-salt")
    derived_str_salt = signer_str_salt.derive_key()
    expected_str_salt = hashlib.sha1(b"custom-salt" + b"signer" + b"test-key").digest()
    assert derived_str_salt == expected_str_salt
```


# LLM-generated content at query #3
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with a basic signer
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    
    # Verify with correct signature
    assert signer.verify_signature(value, sig) is True
    
    # Verify with incorrect signature
    assert signer.verify_signature(value, b"invalid-sig") is False
    
    # Verify with empty signature
    assert signer.verify_signature(value, b"") is False
    
    # Verify with non-base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False
    
    # Test with string value
    value_str = "test-string-value"
    sig_str = signer.get_signature(value_str)
    assert signer.verify_signature(value_str, sig_str) is True
    
    # Test with key rotation - multiple secret keys
    signer_rotation = Signer(["old-key", "new-key"])
    value_rotation = b"rotation-test"
    sig_rotation = signer_rotation.get_signature(value_rotation)
    
    # Should verify with any key in the list
    assert signer_rotation.verify_signature(value_rotation, sig_rotation) is True
    
    # Test with different salt
    signer_salt = Signer("secret-key", salt="custom-salt")
    value_salt = b"salt-test"
    sig_salt = signer_salt.get_signature(value_salt)
    
    # Should verify with same signer
    assert signer_salt.verify_signature(value_salt, sig_salt) is True
    
    # Different salt should fail (different signer instance with same key but different salt)
    signer_salt2 = Signer("secret-key", salt="different-salt")
    assert signer_salt2.verify_signature(value_salt, sig_salt) is False
    
    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value_none = b"none-alg-test"
    sig_none = signer_none.get_signature(value_none)
    assert sig_none == base64_encode(b"")
    assert signer_none.verify_signature(value_none, sig_none) is True
    
    # Test with different digest method
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value_sha256 = b"sha256-test"
    sig_sha256 = signer_sha256.get_signature(value_sha256)
    assert signer_sha256.verify_signature(value_sha256, sig_sha256) is True
    
    # Test with HMAC key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    value_hmac = b"hmac-derivation-test"
    sig_hmac = signer_hmac.get_signature(value_hmac)
    assert signer_hmac.verify_signature(value_hmac, sig_hmac) is True
    
    # Test with concat key derivation
    signer_concat = Signer("secret-key", key_derivation="concat")
    value_concat = b"concat-derivation-test"
    sig_concat = signer_concat.get_signature(value_concat)
    assert signer_concat.verify_signature(value_concat, sig_concat) is True
    
    # Test with none key derivation
    signer_none_derivation = Signer("secret-key", key_derivation="none")
    value_none_derivation = b"none-derivation-test"
    sig_none_derivation = signer_none_derivation.get_signature(value_none_derivation)
    assert signer_none_derivation.verify_signature(value_none_derivation, sig_none_derivation) is True
    
    # Test with bytes and string value types
    signer_types = Signer("secret-key")
    value_bytes = b"bytes-value"
    value_str = "string-value"
    sig_bytes = signer_types.get_signature(value_bytes)
    sig_str = signer_types.get_signature(value_str)
    
    assert signer_types.verify_signature(value_bytes, sig_bytes) is True
    assert signer_types.verify_signature(value_str, sig_str) is True
    
    # Test that signature from bytes works with string value and vice versa
    # (they should be treated as different values since bytes and str differ)
    assert signer_types.verify_signature(value_bytes, sig_str) is False
    assert signer_types.verify_signature(value_str, sig_bytes) is False
```


# LLM-generated content at query #4
#--------------------------

```python
def test_Signer_derive_key():
    signer = Signer("secret-key", salt="test-salt")
    
    # Test with default key derivation (django-concat)
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key
    
    # Test with concat derivation
    signer_concat = Signer("secret-key", salt="test-salt", key_derivation="concat")
    derived_key_concat = signer_concat.derive_key()
    expected_key_concat = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert derived_key_concat == expected_key_concat
    
    # Test with hmac derivation
    signer_hmac = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    derived_key_hmac = signer_hmac.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key_hmac = mac.digest()
    assert derived_key_hmac == expected_key_hmac
    
    # Test with none derivation
    signer_none = Signer("secret-key", salt="test-salt", key_derivation="none")
    derived_key_none = signer_none.derive_key()
    assert derived_key_none == b"secret-key"
    
    # Test with custom secret_key parameter
    derived_key_custom = signer.derive_key("custom-secret")
    expected_key_custom = hashlib.sha1(b"test-salt" + b"signer" + b"custom-secret").digest()
    assert derived_key_custom == expected_key_custom
    
    # Test with bytes secret key
    signer_bytes = Signer(b"secret-key-bytes", salt=b"test-salt-bytes")
    derived_key_bytes = signer_bytes.derive_key()
    expected_key_bytes = hashlib.sha1(b"test-salt-bytes" + b"signer" + b"secret-key-bytes").digest()
    assert derived_key_bytes == expected_key_bytes
    
    # Test with empty salt
    signer_no_salt = Signer("secret-key", salt=None)
    assert signer_no_salt.salt == b"itsdangerous.Signer"
    derived_key_no_salt = signer_no_salt.derive_key()
    expected_key_no_salt = hashlib.sha1(b"itsdangerous.Signer" + b"signer" + b"secret-key").digest()
    assert derived_key_no_salt == expected_key_no_salt
    
    # Test with unknown key derivation raises TypeError
    signer_unknown = Signer("secret-key", key_derivation="unknown")
    try:
        signer_unknown.derive_key()
        assert False, "Expected TypeError"
    except TypeError:
        pass
```


# LLM-generated content at query #5
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid")
    assert signer.verify_signature(value, invalid_sig) is False

    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) is True

    # Test with string value (should be converted to bytes)
    str_value = "string-value"
    sig_str = signer.get_signature(str_value)
    assert signer.verify_signature(str_value, sig_str) is True

    # Test with multiple secret keys (key rotation)
    signer_rotation = Signer(["old-key", "new-key"], salt="rotation-salt")
    value_rotation = b"rotation-value"
    sig_rotation = signer_rotation.get_signature(value_rotation)
    assert signer_rotation.verify_signature(value_rotation, sig_rotation) is True

    # Test with NoneAlgorithm
    none_algorithm_signer = Signer("secret", algorithm=NoneAlgorithm())
    none_sig = none_algorithm_signer.get_signature(b"test")
    assert none_algorithm_signer.verify_signature(b"test", none_sig) is True

    # Test with corrupted signature (base64 decode failure)
    corrupted_sig = b"!@#$%^&*"
    assert signer.verify_signature(b"test", corrupted_sig) is False

    # Test with different algorithms
    hmac_signer = Signer("secret-key", algorithm=HMACAlgorithm())
    hmac_sig = hmac_signer.get_signature(b"hmac-test")
    assert hmac_signer.verify_signature(b"hmac-test", hmac_sig) is True

    # Test with different key derivation methods
    concat_signer = Signer("secret-key", key_derivation="concat")
    concat_sig = concat_signer.get_signature(b"concat-test")
    assert concat_signer.verify_signature(b"concat-test", concat_sig) is True

    django_signer = Signer("secret-key", key_derivation="django-concat")
    django_sig = django_signer.get_signature(b"django-test")
    assert django_signer.verify_signature(b"django-test", django_sig) is True

    hmac_derivation_signer = Signer("secret-key", key_derivation="hmac")
    hmac_derivation_sig = hmac_derivation_signer.get_signature(b"hmac-derivation-test")
    assert hmac_derivation_signer.verify_signature(b"hmac-derivation-test", hmac_derivation_sig) is True

    none_derivation_signer = Signer("secret-key", key_derivation="none")
    none_derivation_sig = none_derivation_signer.get_signature(b"none-derivation-test")
    assert none_derivation_signer.verify_signature(b"none-derivation-test", none_derivation_sig) is True

    # Test with custom digest method
    custom_signer = Signer("secret-key", digest_method=hashlib.sha256)
    custom_sig = custom_signer.get_signature(b"custom-digest")
    assert custom_signer.verify_signature(b"custom-digest", custom_sig) is True

    # Test with different separator
    sep_signer = Signer("secret-key", sep=b"|")
    sep_sig = sep_signer.get_signature(b"sep-test")
    assert sep_signer.verify_signature(b"sep-test", sep_sig) is True
```


# LLM-generated content at query #6
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid-signature")
    assert signer.verify_signature(value, invalid_sig) is False

    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) is True

    # Test with string value (not bytes)
    str_value = "test-string"
    sig_str = signer.get_signature(str_value)
    assert signer.verify_signature(str_value, sig_str) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, "!!!invalid-base64!!!") is False

    # Test with NoneAlgorithm (empty signature)
    none_alg_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value2 = b"test-value-2"
    sig2 = none_alg_signer.get_signature(value2)
    assert none_alg_signer.verify_signature(value2, sig2) is True

    # Test with key rotation (multiple secret keys)
    rotated_signer = Signer(["old-key", "new-key"])
    value3 = b"test-value-3"
    sig3 = rotated_signer.get_signature(value3)
    # Should verify with any key in the list
    assert rotated_signer.verify_signature(value3, sig3) is True

    # Test with different salt
    signer_with_salt = Signer("secret-key", salt="custom-salt")
    value4 = b"test-value-4"
    sig4 = signer_with_salt.get_signature(value4)
    assert signer_with_salt.verify_signature(value4, sig4) is True

    # Test that signature from different salt doesn't verify
    another_signer = Signer("secret-key", salt="different-salt")
    assert another_signer.verify_signature(value4, sig4) is False

    # Test with different key derivation methods
    concat_signer = Signer("secret-key", key_derivation="concat")
    value5 = b"test-value-5"
    sig5 = concat_signer.get_signature(value5)
    assert concat_signer.verify_signature(value5, sig5) is True

    hmac_signer = Signer("secret-key", key_derivation="hmac")
    value6 = b"test-value-6"
    sig6 = hmac_signer.get_signature(value6)
    assert hmac_signer.verify_signature(value6, sig6) is True
```


# LLM-generated content at query #7
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default algorithm (HMAC-SHA1) and single key
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    
    # Valid signature should return True
    assert signer.verify_signature(value, sig) == True
    
    # Invalid signature should return False
    assert signer.verify_signature(value, b"invalid-sig") == False
    
    # Invalid base64 signature should return False without raising exception
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False
    
    # Test with multiple secret keys (key rotation)
    signer_rotation = Signer(
        ["old-key", "new-key"],
        salt="test-salt"
    )
    
    # Sign with newest key (new-key)
    value2 = b"test-value-2"
    sig2 = signer_rotation.get_signature(value2)
    
    # Should verify with any key in the list
    assert signer_rotation.verify_signature(value2, sig2) == True
    
    # Test with NoneAlgorithm
    none_signer = Signer(
        "secret-key",
        algorithm=NoneAlgorithm()
    )
    value3 = b"test-value-3"
    sig3 = none_signer.get_signature(value3)
    assert sig3 == base64_encode(b"")
    assert none_signer.verify_signature(value3, sig3) == True
    
    # Test with string value (should be converted to bytes)
    assert signer.verify_signature("test-value", sig) == True
    
    # Test with string signature (should be converted to bytes)
    assert signer.verify_signature(value, sig.decode()) == True
    
    # Test with empty value
    empty_sig = signer.get_signature(b"")
    assert signer.verify_signature(b"", empty_sig) == True
    
    # Test with NoneAlgorithm and empty value
    assert none_signer.verify_signature(b"", b"") == True
```


# LLM-generated content at query #8
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with string value
    assert signer.verify_signature("test-value", sig) is True
    
    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") is False
    
    # Test with different separator
    signer2 = Signer("secret-key", sep=b"|")
    value2 = b"another-value"
    sig2 = signer2.get_signature(value2)
    assert signer2.verify_signature(value2, sig2) is True
    
    # Test with key rotation - old keys should still verify
    signer3 = Signer(["old-key", "new-key"])
    value3 = b"rotation-test"
    sig3 = signer3.get_signature(value3)
    assert signer3.verify_signature(value3, sig3) is True
    
    # Test with different algorithm
    signer4 = Signer("secret", algorithm=NoneAlgorithm())
    value4 = b"none-algorithm"
    sig4 = signer4.get_signature(value4)
    assert signer4.verify_signature(value4, sig4) is True
    assert sig4 == b""
    
    # Test with different key derivation
    signer5 = Signer("secret", key_derivation="concat")
    value5 = b"concat-derivation"
    sig5 = signer5.get_signature(value5)
    assert signer5.verify_signature(value5, sig5) is True
    
    # Test with different digest method
    signer6 = Signer("secret", digest_method=hashlib.sha256)
    value6 = b"sha256-test"
    sig6 = signer6.get_signature(value6)
    assert signer6.verify_signature(value6, sig6) is True
    
    # Test with salt
    signer7 = Signer("secret", salt=b"custom-salt")
    value7 = b"salted-value"
    sig7 = signer7.get_signature(value7)
    assert signer7.verify_signature(value7, sig7) is True
    
    # Test that base64 decoding errors return False
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False
    
    # Test with wrong key
    signer8 = Signer("different-key")
    assert signer8.verify_signature(value, sig) is False
    
    # Test with empty value
    assert signer.verify_signature(b"", signer.get_signature(b"")) is True
    
    # Test with long value
    long_value = b"x" * 10000
    long_sig = signer.get_signature(long_value)
    assert signer.verify_signature(long_value, long_sig) is True
```


# LLM-generated content at query #9
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default parameters
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with string value
    assert signer.verify_signature("test-value", sig) is True
    
    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False
    
    # Test with empty value
    empty_sig = signer.get_signature(b"")
    assert signer.verify_signature(b"", empty_sig) is True
    
    # Test with key rotation
    signer_rotation = Signer(["old-key", "new-key"])
    value = b"test"
    
    # Sign with newest key
    sig_new = signer_rotation.get_signature(value)
    assert signer_rotation.verify_signature(value, sig_new) is True
    
    # Verify with old key should also work
    # Create a signer with only the old key to get its signature
    old_signer = Signer("old-key", salt=signer_rotation.salt, sep=signer_rotation.sep)
    sig_old = old_signer.get_signature(value)
    assert signer_rotation.verify_signature(value, sig_old) is True
    
    # Test with different salt
    signer2 = Signer("secret-key", salt="different-salt")
    sig2 = signer2.get_signature(value)
    assert signer.verify_signature(value, sig2) is False
    
    # Test with different separator
    signer3 = Signer("secret-key", sep=b"|")
    sig3 = signer3.get_signature(value)
    assert signer3.verify_signature(value, sig3) is True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False
    
    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True
    
    # Test with HMACAlgorithm
    import hashlib
    signer_sha256 = Signer("secret-key", algorithm=HMACAlgorithm(hashlib.sha256))
    sig_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig_sha256) is True
    
    # Test with key derivation methods
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_derivation = Signer("secret-key", key_derivation=derivation)
        sig_derivation = signer_derivation.get_signature(value)
        assert signer_derivation.verify_signature(value, sig_derivation) is True
```


# LLM-generated content at query #10
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True
    
    # Test with string value
    assert signer.verify_signature("test value", sig) == True
    
    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid_sig") == False
    
    # Test with empty value
    empty_sig = signer.get_signature(b"")
    assert signer.verify_signature(b"", empty_sig) == True
    
    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value2 = b"test value 2"
    sig2 = none_signer.get_signature(value2)
    assert none_signer.verify_signature(value2, sig2) == True
    
    # Test with multiple secret keys (key rotation)
    signer_rotation = Signer(["old-key", "new-key"])
    value3 = b"rotation test"
    sig3 = signer_rotation.get_signature(value3)
    # Should verify with both old and new keys
    assert signer_rotation.verify_signature(value3, sig3) == True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid base64!!!") == False
    
    # Test with modified value but correct signature
    sig4 = signer.get_signature(b"original value")
    assert signer.verify_signature(b"modified value", sig4) == False
    
    # Test with different salt
    signer2 = Signer("secret-key", salt=b"different-salt")
    sig5 = signer2.get_signature(value)
    assert signer2.verify_signature(value, sig5) == True
    # Should not verify with original signer
    assert signer.verify_signature(value, sig5) == False
```


# LLM-generated content at query #11
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with modified value
    modified_value = b"modified-value"
    assert signer.verify_signature(modified_value, sig) is False

    # Test with empty value
    empty_value = b""
    empty_sig = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, empty_sig) is True

    # Test with different salt
    signer2 = Signer("secret-key", salt="different-salt")
    sig2 = signer2.get_signature(value)
    assert signer.verify_signature(value, sig2) is False

    # Test with different key derivation
    signer3 = Signer("secret-key", key_derivation="concat")
    sig3 = signer3.get_signature(value)
    assert signer.verify_signature(value, sig3) is True
    assert signer.verify_signature(value, sig) is False

    # Test with key rotation (multiple keys)
    signer4 = Signer(["old-key", "new-key"])
    old_sig = signer4.get_signature(value)
    assert signer4.verify_signature(value, old_sig) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False

    # Test with string value (not bytes)
    assert signer.verify_signature("test-value", sig) is True

    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) is True
```


# LLM-generated content at query #12
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    
    # Test valid signature
    assert signer.verify_signature(value, sig) == True
    
    # Test invalid signature
    assert signer.verify_signature(value, b"invalid-sig") == False
    
    # Test with string value
    assert signer.verify_signature("test-value", sig) == True
    
    # Test with string signature
    sig_str = sig.decode('utf-8')
    assert signer.verify_signature(value, sig_str) == True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") == False
    
    # Test with key rotation (multiple secret keys)
    signer_rotation = Signer(["old-key", "new-key"])
    value2 = b"test-value-2"
    sig2 = signer_rotation.get_signature(value2)
    
    # Verify with newest key
    assert signer_rotation.verify_signature(value2, sig2) == True
    
    # Verify with each key in the list
    for key in signer_rotation.secret_keys:
        derived_key = signer_rotation.derive_key(key)
        sig_from_key = signer_rotation.algorithm.get_signature(derived_key, value2)
        encoded_sig = base64_encode(sig_from_key)
        assert signer_rotation.verify_signature(value2, encoded_sig) == True
    
    # Test with different salt
    signer_diff_salt = Signer("secret-key", salt="different-salt")
    value3 = b"test-value-3"
    sig3 = signer_diff_salt.get_signature(value3)
    assert signer_diff_salt.verify_signature(value3, sig3) == True
    
    # Signature from different salt should not verify
    assert signer.verify_signature(value3, sig3) == False
    
    # Test with custom separator
    signer_custom_sep = Signer("secret-key", sep=b"|")
    value4 = b"test-value-4"
    sig4 = signer_custom_sep.get_signature(value4)
    assert signer_custom_sep.verify_signature(value4, sig4) == True
    
    # Test with NoneAlgorithm
    signer_none_alg = Signer("secret-key", algorithm=NoneAlgorithm())
    value5 = b"test-value-5"
    sig5 = signer_none_alg.get_signature(value5)
    assert signer_none_alg.verify_signature(value5, sig5) == True
    # With NoneAlgorithm, any signature should verify (since empty signature returned)
    assert signer_none_alg.verify_signature(value5, b"anything") == True
```


# LLM-generated content at query #13
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default parameters
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True
    
    # Test with string value
    sig_str = signer.get_signature("test value")
    assert signer.verify_signature("test value", sig_str) == True
    
    # Test with wrong signature
    assert signer.verify_signature(value, b"wrong_signature") == False
    
    # Test with empty value
    sig_empty = signer.get_signature(b"")
    assert signer.verify_signature(b"", sig_empty) == True
    
    # Test with different separator
    signer2 = Signer("secret-key", sep=b"|")
    value2 = b"another value"
    sig2 = signer2.get_signature(value2)
    assert signer2.verify_signature(value2, sig2) == True
    
    # Test with key rotation (multiple secret keys)
    signer3 = Signer(["old-key", "new-key"])
    value3 = b"rotated key value"
    sig3 = signer3.get_signature(value3)
    # Should verify with any key in the list
    assert signer3.verify_signature(value3, sig3) == True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid_base64!!!") == False
    
    # Test with different algorithm
    none_algorithm = NoneAlgorithm()
    signer4 = Signer("secret-key", algorithm=none_algorithm)
    value4 = b"none algorithm test"
    sig4 = signer4.get_signature(value4)
    assert signer4.verify_signature(value4, sig4) == True
    
    # Test with HMAC algorithm and different digest method
    import hashlib
    signer5 = Signer("secret-key", digest_method=hashlib.sha256)
    value5 = b"sha256 test"
    sig5 = signer5.get_signature(value5)
    assert signer5.verify_signature(value5, sig5) == True
    
    # Test with salt
    signer6 = Signer("secret-key", salt=b"custom-salt")
    value6 = b"salted value"
    sig6 = signer6.get_signature(value6)
    assert signer6.verify_signature(value6, sig6) == True
```


# LLM-generated content at query #14
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid") is False
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") is False
    
    # Test with non-base64 signature
    assert signer.verify_signature(value, b"!!!") is False
    
    # Test with string value
    assert signer.verify_signature("test value", sig) is True
    
    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) is True
    
    # Test with key rotation - verify with older key
    signer_rotation = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test value"
    # Sign with new key (last in list)
    sig = signer_rotation.get_signature(value)
    # Verify with both keys available
    assert signer_rotation.verify_signature(value, sig) is True
    
    # Test with key rotation - verify signature signed with older key
    old_signer = Signer("old-key", salt="test-salt")
    old_sig = old_signer.get_signature(value)
    assert signer_rotation.verify_signature(value, old_sig) is True
```


# LLM-generated content at query #15
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("test-secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    
    # Verify with correct signature
    assert signer.verify_signature(value, signature) == True
    
    # Verify with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") == False
    
    # Test with string value
    assert signer.verify_signature("test-value", signature) == True
    
    # Test with key rotation
    signer_rotation = Signer(["old-key", "new-key"])
    value = b"rotation-test"
    signature = signer_rotation.get_signature(value)
    
    # Verify with newest key (used for signing)
    assert signer_rotation.verify_signature(value, signature) == True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False
    
    # Test with empty value
    assert signer.verify_signature(b"", signer.get_signature(b"")) == True
    
    # Test with custom separator
    custom_signer = Signer("test-secret", sep=b"|")
    value = b"custom-sep"
    signature = custom_signer.get_signature(value)
    assert custom_signer.verify_signature(value, signature) == True
    
    # Test with different salt
    signer_custom_salt = Signer("test-secret", salt=b"custom-salt")
    value = b"salt-test"
    signature = signer_custom_salt.get_signature(value)
    assert signer_custom_salt.verify_signature(value, signature) == True
    
    # Verify that signature from one salt doesn't work with another
    signer_different_salt = Signer("test-secret", salt=b"different-salt")
    assert signer_different_salt.verify_signature(value, signature) == False
    
    # Test with HMAC key derivation
    signer_hmac = Signer("test-secret", key_derivation="hmac")
    value = b"hmac-test"
    signature = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, signature) == True
    
    # Test with concat key derivation
    signer_concat = Signer("test-secret", key_derivation="concat")
    value = b"concat-test"
    signature = signer_concat.get_signature(value)
    assert signer_concat.verify_signature(value, signature) == True
    
    # Test with NoneAlgorithm
    signer_none = Signer("test-secret", algorithm=NoneAlgorithm())
    value = b"none-algorithm"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) == True
    
    # Test with bytes secret key
    signer_bytes = Signer(b"bytes-secret-key")
    value = b"bytes-test"
    signature = signer_bytes.get_signature(value)
    assert signer_bytes.verify_signature(value, signature) == True
```


# LLM-generated content at query #16
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    
    # Valid signature should return True
    assert signer.verify_signature(value, sig) == True
    
    # Invalid signature should return False
    assert signer.verify_signature(value, b"invalid-sig") == False
    
    # Empty signature should return False
    assert signer.verify_signature(value, b"") == False
    
    # Non-base64 signature should return False
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False
    
    # Test with string value
    assert signer.verify_signature("test-value", sig) == True
    
    # Test with string signature
    assert signer.verify_signature(value, sig.decode()) == True
    
    # Test with different salt
    signer2 = Signer("secret-key", salt="different-salt")
    sig2 = signer2.get_signature(value)
    # Signature from different salt should not match
    assert signer.verify_signature(value, sig2) == False
    
    # Test with key rotation - verify with older key
    signer3 = Signer(["old-key", "new-key"])
    sig_old = signer3.get_signature(value)  # signed with new-key
    assert signer3.verify_signature(value, sig_old) == True
    
    # Test with NoneAlgorithm
    signer4 = Signer("secret-key", algorithm=NoneAlgorithm())
    sig4 = signer4.get_signature(value)
    assert signer4.verify_signature(value, sig4) == True
    # With NoneAlgorithm, any signature should be valid
    assert signer4.verify_signature(value, b"anything") == False  # base64 decode fails
    
    # Test verify_signature returns False for empty base64 decoded signature
    # (NoneAlgorithm returns empty signature)
    signer5 = Signer("secret-key", algorithm=NoneAlgorithm())
    empty_sig = base64_encode(b"")
    assert signer5.verify_signature(value, empty_sig) == True
    
    # Test with HMACAlgorithm custom digest
    import hashlib
    signer6 = Signer("secret-key", digest_method=hashlib.sha256)
    sig6 = signer6.get_signature(value)
    assert signer6.verify_signature(value, sig6) == True
    assert signer6.verify_signature(value, b"wrong") == False
    
    # Test with bytes value containing separator
    signer7 = Signer("test-key", sep=b"|")
    value_with_sep = b"test|value"
    sig7 = signer7.get_signature(value_with_sep)
    assert signer7.verify_signature(value_with_sep, sig7) == True
```


# LLM-generated content at query #17
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default parameters
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    
    # Valid signature should return True
    assert signer.verify_signature(value, sig) == True
    
    # Invalid signature should return False
    assert signer.verify_signature(value, b"invalid-sig") == False
    
    # Empty bytes signature should return False
    assert signer.verify_signature(value, b"") == False
    
    # Test with string value
    assert signer.verify_signature("test-value", sig) == True
    
    # Test with different separator
    signer2 = Signer("secret-key", sep=b"|")
    value2 = b"another-value"
    sig2 = signer2.get_signature(value2)
    assert signer2.verify_signature(value2, sig2) == True
    
    # Test with key rotation (multiple secret keys)
    signer3 = Signer(["old-key", "new-key"])
    value3 = b"key-rotation-test"
    sig3 = signer3.get_signature(value3)
    
    # Verify with newest key (last in list)
    assert signer3.verify_signature(value3, sig3) == True
    
    # Create a signer that only knows the old key
    signer_old = Signer("old-key")
    # The signature created with new key should not verify with old key
    assert signer_old.verify_signature(value3, sig3) == False
    
    # Test with salt parameter
    signer4 = Signer("secret-key", salt=b"custom-salt")
    value4 = b"salted-value"
    sig4 = signer4.get_signature(value4)
    assert signer4.verify_signature(value4, sig4) == True
    
    # Different salt should produce different signature
    signer5 = Signer("secret-key", salt=b"different-salt")
    assert signer5.verify_signature(value4, sig4) == False
    
    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value5 = b"none-algorithm"
    sig5 = signer_none.get_signature(value5)
    # NoneAlgorithm produces empty signature
    assert sig5 == b""
    assert signer_none.verify_signature(value5, sig5) == True
    
    # Test with HMACAlgorithm explicitly
    import hashlib
    hmac_algo = HMACAlgorithm(hashlib.sha256)
    signer_hmac = Signer("secret-key", algorithm=hmac_algo)
    value6 = b"hmac-sha256"
    sig6 = signer_hmac.get_signature(value6)
    assert signer_hmac.verify_signature(value6, sig6) == True
    
    # Test with corrupted signature
    assert signer.verify_signature(value, sig + b"corrupt") == False
    
    # Test with different value
    assert signer.verify_signature(b"different-value", sig) == False
    
    # Test with non-bytes sig that can't be base64 decoded
    assert signer.verify_signature(value, "!!invalid-base64!!") == False
```


# LLM-generated content at query #18
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default parameters
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with string value
    assert signer.verify_signature("test-value", sig) is True

    # Test with wrong signature
    wrong_sig = base64_encode(b"wrong-signature")
    assert signer.verify_signature(value, wrong_sig) is False

    # Test with empty signature
    assert signer.verify_signature(value, b"") is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False

    # Test with multiple secret keys (key rotation)
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with different salt
    signer1 = Signer("secret-key", salt="salt1")
    signer2 = Signer("secret-key", salt="salt2")
    value = b"test-value"
    sig1 = signer1.get_signature(value)
    assert signer1.verify_signature(value, sig1) is True
    assert signer2.verify_signature(value, sig1) is False

    # Test with different separator
    signer = Signer("secret-key", sep=b"|")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    assert signer.verify_signature(value, b"") is True

    # Test with HMACAlgorithm and custom digest method
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with different key derivation
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with empty value
    signer = Signer("secret-key")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with bytes value containing separator
    signer = Signer("secret-key")
    value = b"test.value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
```


# LLM-generated content at query #19
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with string value
    assert signer.verify_signature("test-value", sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with empty signature
    assert signer.verify_signature(value, b"") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True

    # Test with key rotation (multiple secret keys)
    signer_rotation = Signer(["old-key", "new-key"])
    # Sign with newest key
    sig_new = signer_rotation.get_signature(value)
    assert signer_rotation.verify_signature(value, sig_new) is True

    # Test with different salt
    signer_diff_salt = Signer("secret-key", salt="different-salt")
    sig_diff = signer_diff_salt.get_signature(value)
    assert signer.verify_signature(value, sig_diff) is False
    assert signer_diff_salt.verify_signature(value, sig_diff) is True

    # Test with different key derivation methods
    signer_concat = Signer("secret-key", key_derivation="concat")
    sig_concat = signer_concat.get_signature(value)
    assert signer_concat.verify_signature(value, sig_concat) is True
    assert signer.verify_signature(value, sig_concat) is False

    signer_hmac = Signer("secret-key", key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac) is True
    assert signer.verify_signature(value, sig_hmac) is False

    signer_none_derivation = Signer("secret-key", key_derivation="none")
    sig_none_derivation = signer_none_derivation.get_signature(value)
    assert signer_none_derivation.verify_signature(value, sig_none_derivation) is True
    assert signer.verify_signature(value, sig_none_derivation) is False
```


# LLM-generated content at query #20
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid") is False
    
    # Test with string value
    assert signer.verify_signature("test value", sig) is True
    
    # Test with empty value
    empty_sig = signer.get_signature(b"")
    assert signer.verify_signature(b"", empty_sig) is True
    
    # Test with base64 decode exception (invalid base64)
    assert signer.verify_signature(value, b"!!!invalid!!!") is False
    
    # Test with key rotation (multiple keys)
    signer_rotation = Signer(["old-key", "new-key"])
    value2 = b"test value 2"
    sig2 = signer_rotation.get_signature(value2)
    # Verify with each key
    assert signer_rotation.verify_signature(value2, sig2) is True
    
    # Test with custom algorithm (NoneAlgorithm)
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value3 = b"test value 3"
    sig3 = signer_none.get_signature(value3)
    assert signer_none.verify_signature(value3, sig3) is True
    
    # Test with custom separator
    signer_custom_sep = Signer("secret-key", sep=b"|")
    value4 = b"test value 4"
    sig4 = signer_custom_sep.get_signature(value4)
    assert signer_custom_sep.verify_signature(value4, sig4) is True
    
    # Test with custom key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    value5 = b"test value 5"
    sig5 = signer_hmac.get_signature(value5)
    assert signer_hmac.verify_signature(value5, sig5) is True
    
    # Test signature verification fails with wrong value
    assert signer.verify_signature(b"wrong value", sig) is False
    
    # Test with bytes signature
    assert signer.verify_signature(value, sig) is True
    
    # Test signature from different key fails
    other_signer = Signer("different-secret")
    other_sig = other_signer.get_signature(value)
    assert signer.verify_signature(value, other_sig) is False
    
    # Test with non-ascii value
    non_ascii_value = b"test value with \xc3\xa9"
    non_ascii_sig = signer.get_signature(non_ascii_value)
    assert signer.verify_signature(non_ascii_value, non_ascii_sig) is True
```


# LLM-generated content at query #21
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True
    
    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid")
    assert signer.verify_signature(value, invalid_sig) == False
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") == False
    
    # Test with different value
    wrong_value = b"wrong-value"
    assert signer.verify_signature(wrong_value, sig) == False
    
    # Test with key rotation - all keys should verify
    signer_rotation = Signer(["old-key", "new-key"])
    value2 = b"rotation-test"
    sig2 = signer_rotation.get_signature(value2)
    assert signer_rotation.verify_signature(value2, sig2) == True
    
    # Test with string input for value and sig
    assert signer.verify_signature("test-value", sig) == True
    
    # Test with NoneAlgorithm
    none_signer = Signer("key", algorithm=NoneAlgorithm())
    value3 = b"none-test"
    sig3 = none_signer.get_signature(value3)
    assert none_signer.verify_signature(value3, sig3) == True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False
```


# LLM-generated content at query #22
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with empty string
    empty_value = b""
    empty_sig = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, empty_sig) is True

    # Test with bytes and str inputs
    assert signer.verify_signature(b"test", sig) is False  # different value
    assert signer.verify_signature("test-value", sig) is True  # str value works

    # Test with corrupted signature (base64 decode failure)
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False

    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) is True

    # Test with key rotation (multiple secret keys)
    rotated_signer = Signer(
        ["old-key", "new-key"],
        salt="test-salt"
    )
    # Sign with new key
    rotated_sig = rotated_signer.get_signature(value)
    assert rotated_signer.verify_signature(value, rotated_sig) is True

    # Test with different digest method
    sha256_signer = Signer("secret-key", digest_method=hashlib.sha256)
    sha256_sig = sha256_signer.get_signature(value)
    assert sha256_signer.verify_signature(value, sha256_sig) is True
    assert sha256_signer.verify_signature(value, b"wrong-sig") is False

    # Test with different key derivation
    hmac_derivation_signer = Signer(
        "secret-key",
        key_derivation="hmac"
    )
    hmac_sig = hmac_derivation_signer.get_signature(value)
    assert hmac_derivation_signer.verify_signature(value, hmac_sig) is True

    # Test with "none" key derivation
    none_derivation_signer = Signer(
        "secret-key",
        key_derivation="none"
    )
    none_derivation_sig = none_derivation_signer.get_signature(value)
    assert none_derivation_signer.verify_signature(value, none_derivation_sig) is True
```


# LLM-generated content at query #23
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings (HMAC-SHA1, django-concat key derivation)
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature (wrong bytes)
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with invalid signature (wrong base64)
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False

    # Test with empty value
    sig_empty = signer.get_signature(b"")
    assert signer.verify_signature(b"", sig_empty) is True

    # Test with different salt
    signer2 = Signer("secret-key", salt=b"different-salt")
    sig2 = signer2.get_signature(value)
    assert signer2.verify_signature(value, sig2) is True
    assert signer.verify_signature(value, sig2) is False  # Different salt should fail

    # Test with different separator
    signer3 = Signer("secret-key", sep=b":")
    sig3 = signer3.get_signature(value)
    assert signer3.verify_signature(value, sig3) is True
    assert signer.verify_signature(value, sig3) is False  # Different sep should fail

    # Test with multiple secret keys (key rotation)
    signer4 = Signer(["old-key", "new-key"])
    sig4 = signer4.get_signature(value)
    assert signer4.verify_signature(value, sig4) is True
    # Old key should also work for verification
    old_signer = Signer("old-key")
    old_sig = old_signer.get_signature(value)
    assert signer4.verify_signature(value, old_sig) is True

    # Test with string value (not bytes)
    assert signer.verify_signature("test-value", sig) is True

    # Test with string signature (not bytes)
    sig_str = sig.decode('ascii')
    assert signer.verify_signature(value, sig_str) is True

    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, sig_none) is True
    # Any signature should pass with NoneAlgorithm
    assert none_signer.verify_signature(value, b"anything") is True

    # Test with different key_derivation methods
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        s = Signer("secret-key", key_derivation=derivation)
        sig_d = s.get_signature(value)
        assert s.verify_signature(value, sig_d) is True

    # Test with different digest methods
    s_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    sig_sha256 = s_sha256.get_signature(value)
    assert s_sha256.verify_signature(value, sig_sha256) is True
    # SHA1 signer should not verify SHA256 signature
    assert signer.verify_signature(value, sig_sha256) is False

    # Edge cases
    # Test with very long value
    long_value = b"a" * 10000
    sig_long = signer.get_signature(long_value)
    assert signer.verify_signature(long_value, sig_long) is True

    # Test with binary value containing null bytes
    binary_value = b"\x00\x01\x02\xff"
    sig_bin = signer.get_signature(binary_value)
    assert signer.verify_signature(binary_value, sig_bin) is True
```


# LLM-generated content at query #24
#--------------------------

```python
def test_Signer_verify_signature():
    """Test Signer.verify_signature method."""
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid")
    assert signer.verify_signature(value, invalid_sig) is False
    
    # Test with corrupted signature
    corrupted_sig = sig + b"x"
    assert signer.verify_signature(value, corrupted_sig) is False
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") is False
    
    # Test with string value
    assert signer.verify_signature("test-value", sig) is True
    
    # Test with different key
    signer2 = Signer("different-key")
    sig2 = signer2.get_signature(value)
    assert signer.verify_signature(value, sig2) is False
    
    # Test with key rotation (verifying with older key)
    signer3 = Signer(["old-key", "new-key"])
    old_signer = Signer("old-key")
    old_sig = old_signer.get_signature(value)
    assert signer3.verify_signature(value, old_sig) is True
    
    # Test with NoneAlgorithm
    none_signer = Signer("key", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) is True
    
    # Test with base64 decode failure (invalid characters)
    invalid_base64 = b"!!!invalid!!!"
    assert signer.verify_signature(value, invalid_base64) is False
```


# LLM-generated content at query #25
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with a simple string value and its signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with wrong signature
    wrong_sig = base64_encode(b"wrong-signature")
    assert signer.verify_signature(value, wrong_sig) is False
    
    # Test with empty signature
    assert signer.verify_signature(value, b"") is False
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!!!") is False
    
    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) is True
    
    # Test with string value (not bytes)
    string_value = "test-string"
    sig_string = signer.get_signature(string_value)
    assert signer.verify_signature(string_value, sig_string) is True
    
    # Test with key rotation - verify with older key
    signer_with_keys = Signer(["old-key", "new-key"])
    value2 = b"rotate-test"
    sig2 = signer_with_keys.get_signature(value2)
    # Should verify with the last key
    assert signer_with_keys.verify_signature(value2, sig2) is True
    
    # Test with NoneAlgorithm
    none_signer = Signer("test", algorithm=NoneAlgorithm())
    value3 = b"none-alg-test"
    sig3 = none_signer.get_signature(value3)
    assert none_signer.verify_signature(value3, sig3) is True
    
    # Test with different salt
    signer_diff_salt = Signer("secret-key", salt=b"different-salt")
    value4 = b"salt-test"
    sig4 = signer_diff_salt.get_signature(value4)
    assert signer_diff_salt.verify_signature(value4, sig4) is True
    
    # Test that signature from one salt doesn't work with another
    assert signer.verify_signature(value4, sig4) is False
```


# LLM-generated content at query #26
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid-signature")
    assert signer.verify_signature(value, invalid_sig) == False

    # Test with empty signature
    assert signer.verify_signature(value, b"") == False

    # Test with non-base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False

    # Test with different value
    sig = signer.get_signature(b"different-value")
    assert signer.verify_signature(value, sig) == False

    # Test with string value
    assert signer.verify_signature("test-value", sig) == False

    # Test with key rotation - only newest key should verify
    signer = Signer(["old-key", "new-key"])
    value = b"test"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True  # Newest key works
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) == False

    # Test with bytes input for both value and sig
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

    # Test with string input for value
    sig = signer.get_signature(b"test")
    assert signer.verify_signature("test", sig) == True

    # Test with empty value
    sig = signer.get_signature(b"")
    assert signer.verify_signature(b"", sig) == True
    assert signer.verify_signature(b"", b"") == False
```


# LLM-generated content at query #27
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default settings
    signer = Signer("secret-key", salt="test-salt")
    value = b"test value"
    sig = signer.get_signature(value)
    
    # Test valid signature
    assert signer.verify_signature(value, sig) == True
    
    # Test invalid signature
    assert signer.verify_signature(value, b"invalid") == False
    
    # Test with string value
    assert signer.verify_signature("test value", sig) == True
    
    # Test with modified value
    modified_value = b"modified value"
    assert signer.verify_signature(modified_value, sig) == False
    
    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) == True
    
    # Test with key rotation
    signer_rotation = Signer(["old-key", "new-key"])
    value = b"test value"
    sig = signer_rotation.get_signature(value)
    # Old key should still verify
    assert signer_rotation.verify_signature(value, sig) == True
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid base64!!!") == False
    
    # Test with empty value
    empty_value = b""
    sig = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig) == True
```


# LLM-generated content at query #28
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with default parameters
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    
    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False
    
    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False
    
    # Test with string value
    assert signer.verify_signature("test-value", sig) is True
    
    # Test with string signature
    sig_str = sig.decode("ascii")
    assert signer.verify_signature(value, sig_str) is True
    
    # Test with multiple secret keys (key rotation)
    signer2 = Signer(["old-key", "new-key"])
    value2 = b"test-value-2"
    sig2 = signer2.get_signature(value2)
    # Should verify with the newest key
    assert signer2.verify_signature(value2, sig2) is True
    
    # Test that old key still works for verification
    old_signer = Signer("old-key")
    old_sig = old_signer.get_signature(value2)
    assert signer2.verify_signature(value2, old_sig) is True
    
    # Test with empty value
    signer3 = Signer("another-key")
    empty_sig = signer3.get_signature(b"")
    assert signer3.verify_signature(b"", empty_sig) is True
    
    # Test with NoneAlgorithm (no signing)
    signer4 = Signer("key", algorithm=NoneAlgorithm())
    none_sig = signer4.get_signature(b"test")
    assert signer4.verify_signature(b"test", none_sig) is True


# LLM-generated content at query #29
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid") is False

    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) is True

    # Test with different key rotation (multiple secret keys)
    signer_rotation = Signer(["old-key", "new-key"])
    value = b"rotation test"
    sig = signer_rotation.get_signature(value)
    assert signer_rotation.verify_signature(value, sig) is True

    # Test that old keys still verify signatures
    old_signer = Signer("old-key")
    old_sig = old_signer.get_signature(value)
    assert signer_rotation.verify_signature(value, old_sig) is True

    # Test with string value
    str_value = "string value"
    sig_str = signer.get_signature(str_value)
    assert signer.verify_signature(str_value, sig_str) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") is False

    # Test with NoneAlgorithm
    none_signer = Signer("key", algorithm=NoneAlgorithm())
    value = b"none algo"
    sig_none = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, sig_none) is True
    assert none_signer.verify_signature(value, b"") is True
    assert none_signer.verify_signature(value, b"anything") is False

    # Test with different salt
    signer_salt = Signer("key", salt="different-salt")
    value = b"salt test"
    sig_salt = signer_salt.get_signature(value)
    assert signer_salt.verify_signature(value, sig_salt) is True

    # Test that same key with different salt produces different signature
    assert signer.verify_signature(value, sig_salt) is False
```


# LLM-generated content at query #30
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

    # Test with invalid signature
    invalid_sig = base64_encode(b"invalid-signature")
    assert signer.verify_signature(value, invalid_sig) == False

    # Test with empty value
    empty_value = b""
    sig_empty = signer.get_signature(empty_value)
    assert signer.verify_signature(empty_value, sig_empty) == True

    # Test with different secret key
    signer2 = Signer("different-secret")
    sig2 = signer2.get_signature(value)
    assert signer.verify_signature(value, sig2) == False

    # Test with string value
    str_value = "string-value"
    sig_str = signer.get_signature(str_value)
    assert signer.verify_signature(str_value, sig_str) == True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, "not-base64!!") == False

    # Test with empty signature
    assert signer.verify_signature(value, b"") == False

    # Test with NoneAlgorithm
    none_signer = Signer("secret", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) == True

    # Test with key rotation (multiple secret keys)
    rotated_signer = Signer(["old-key", "new-key"])
    old_sig = Signer("old-key").get_signature(value)
    assert rotated_signer.verify_signature(value, old_sig) == True

    new_sig = Signer("new-key").get_signature(value)
    assert rotated_signer.verify_signature(value, new_sig) == True

    # Test with key that is not in rotation
    unknown_sig = Signer("unknown-key").get_signature(value)
    assert rotated_signer.verify_signature(value, unknown_sig) == False

    # Test with different salt
    salt_signer = Signer("secret", salt=b"custom-salt")
    sig_salt = salt_signer.get_signature(value)
    assert salt_signer.verify_signature(value, sig_salt) == True

    # Test that signature from different salt doesn't match
    assert signer.verify_signature(value, sig_salt) == False
```


