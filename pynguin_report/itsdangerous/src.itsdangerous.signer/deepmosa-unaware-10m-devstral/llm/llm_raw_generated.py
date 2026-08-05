####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Signer_derive_key():
    # Test default key derivation (django-concat)
    signer = Signer("secret-key")
    expected_key = _lazy_sha1(b"itsdangerous.Signer" + b"signer" + b"secret-key").digest()
    assert signer.derive_key() == expected_key

    # Test concat key derivation
    signer = Signer("secret-key", key_derivation="concat")
    expected_key = _lazy_sha1(b"itsdangerous.Signer" + b"secret-key").digest()
    assert signer.derive_key() == expected_key

    # Test hmac key derivation
    signer = Signer("secret-key", key_derivation="hmac")
    mac = hmac.new(b"secret-key", digestmod=_lazy_sha1)
    mac.update(b"itsdangerous.Signer")
    expected_key = mac.digest()
    assert signer.derive_key() == expected_key

    # Test none key derivation
    signer = Signer("secret-key", key_derivation="none")
    assert signer.derive_key() == b"secret-key"

    # Test with custom salt
    signer = Signer("secret-key", salt="custom-salt")
    expected_key = _lazy_sha1(b"custom-salt" + b"signer" + b"secret-key").digest()
    assert signer.derive_key() == expected_key

    # Test with specific secret_key parameter
    signer = Signer("secret-key")
    assert signer.derive_key("other-key") == _lazy_sha1(b"itsdangerous.Signer" + b"signer" + b"other-key").digest()

    # Test with bytes secret key
    signer = Signer(b"secret-key")
    expected_key = _lazy_sha1(b"itsdangerous.Signer" + b"signer" + b"secret-key").digest()
    assert signer.derive_key() == expected_key

    # Test with key rotation (multiple keys)
    signer = Signer(["old-key", "new-key"])
    # Should use the newest key by default
    expected_key = _lazy_sha1(b"itsdangerous.Signer" + b"signer" + b"new-key").digest()
    assert signer.derive_key() == expected_key
    # Can specify which key to use
    assert signer.derive_key("old-key") == _lazy_sha1(b"itsdangerous.Signer" + b"signer" + b"old-key").digest()

    # Test invalid key derivation method
    signer = Signer("secret-key", key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer.derive_key()


# LLM-generated content at query #2
#--------------------------

```python
def test_Signer_unsign():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signed_value = signer.sign(value)
    assert signer.unsign(signed_value) == value

    # Test with incorrect signature
    with pytest.raises(BadSignature):
        signer.unsign(b"test-value.incorrect-signature")

    # Test with missing separator
    with pytest.raises(BadSignature):
        signer.unsign(b"test-value")

    # Test with multiple separators
    signer_sep = Signer("secret-key", sep=b"|")
    signed_value_sep = signer_sep.sign(b"test|value")
    assert signer_sep.unsign(signed_value_sep) == b"test|value"

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    signed_value_old = Signer("old-key").sign(b"test-value")
    signed_value_new = Signer("new-key").sign(b"test-value")
    assert signer_rotated.unsign(signed_value_old) == b"test-value"
    assert signer_rotated.unsign(signed_value_new) == b"test-value"

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    signed_value_hmac = signer_hmac.sign(b"test-value")
    assert signer_hmac.unsign(signed_value_hmac) == b"test-value"

    # Test with different digest method
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    signed_value_sha256 = signer_sha256.sign(b"test-value")
    assert signer_sha256.unsign(signed_value_sha256) == b"test-value"

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    signed_value_none = signer_none.sign(b"test-value")
    assert signer_none.unsign(signed_value_none) == b"test-value"


# LLM-generated content at query #3
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) > 0

    # Test with concat key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) > 0

    # Test with hmac key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) > 0

    # Test with none key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key("another-secret")
    assert isinstance(key, bytes)
    assert len(key) > 0

    # Test with invalid key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="invalid")
    with pytest.raises(TypeError):
        signer.derive_key()


# LLM-generated content at query #4
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with wrong value
    assert signer.verify_signature(b"wrong-value", sig) is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    sig_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation
    signer_hmac = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac) is True
    assert signer.verify_signature(value, sig_hmac) is False

    # Test with different digest method
    signer_sha256 = Signer("secret-key", salt="test-salt", digest_method=hashlib.sha256)
    sig_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig_sha256) is True
    assert signer.verify_signature(value, sig_sha256) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True
    assert signer.verify_signature(value, sig_none) is False


# LLM-generated content at query #5
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"invalid-sig")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with key rotation (old key)
    old_key = b"old-secret-key"
    new_key = b"new-secret-key"
    signer = Signer([old_key, new_key])
    value = b"test-value"
    old_sig = hmac.new(old_key, msg=value, digestmod=hashlib.sha1).digest()
    old_sig = base64_encode(old_sig)
    assert signer.verify_signature(value, old_sig) is True

    # Test with key rotation (new key)
    new_sig = hmac.new(new_key, msg=value, digestmod=hashlib.sha1).digest()
    new_sig = base64_encode(new_sig)
    assert signer.verify_signature(value, new_sig) is True

    # Test with key rotation (invalid key)
    invalid_key = b"invalid-secret-key"
    invalid_sig = hmac.new(invalid_key, msg=value, digestmod=hashlib.sha1).digest()
    invalid_sig = base64_encode(invalid_sig)
    assert signer.verify_signature(value, invalid_sig) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with different digest methods
    for digest_method in [hashlib.sha1, hashlib.sha256, hashlib.sha512]:
        signer = Signer("secret-key", digest_method=digest_method)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with empty signature
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    assert signer.verify_signature(value, b"") is True

    # Test with empty value
    signer = Signer("secret-key")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with string value and signature
    signer = Signer("secret-key")
    value = "test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True


# LLM-generated content at query #6
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"wrong-signature")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation (multiple keys)
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)  # Uses newest key
    # Manually create signature with old key
    old_key = signer_rotated.derive_key(secret_key=b"old-key")
    sig_old = signer_rotated.algorithm.get_signature(old_key, value)
    base64_sig_old = base64_encode(sig_old)
    assert signer_rotated.verify_signature(value, base64_sig_old) is True
    assert signer_rotated.verify_signature(value, signature_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", salt="test-salt", digest_method=hashlib.sha256)
    value = b"test-value"
    signature = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True


# LLM-generated content at query #7
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    secret_key = b"secret"
    signer = Signer(secret_key)
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    wrong_sig = b"wrong_signature"
    assert signer.verify_signature(value, wrong_sig) is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"test_sig")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    invalid_base64_sig = b"invalid_base64"
    assert signer.verify_signature(value, invalid_base64_sig) is False

    # Test with multiple secret keys (key rotation)
    secret_keys = [b"old_secret", b"new_secret"]
    signer_rotated = Signer(secret_keys)
    value_rotated = b"test_value_rotated"
    sig_rotated = signer_rotated.get_signature(value_rotated)
    assert signer_rotated.verify_signature(value_rotated, sig_rotated) is True

    # Test with old key still valid
    old_signer = Signer(b"old_secret")
    old_sig = old_signer.get_signature(value_rotated)
    assert signer_rotated.verify_signature(value_rotated, old_sig) is True

    # Test with non-existent key
    non_existent_sig = b"non_existent_sig"
    assert signer_rotated.verify_signature(value_rotated, non_existent_sig) is False


# LLM-generated content at query #8
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-sig") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid-base64") is False

    # Test with key rotation (old key)
    old_key = b"old-secret-key"
    new_key = b"new-secret-key"
    signer = Signer([old_key, new_key], salt="test-salt")
    value = b"test-value"
    sig_old = Signer(old_key, salt="test-salt").get_signature(value)
    assert signer.verify_signature(value, sig_old) is True

    # Test with key rotation (new key)
    sig_new = Signer(new_key, salt="test-salt").get_signature(value)
    assert signer.verify_signature(value, sig_new) is True

    # Test with key rotation (non-existent key)
    sig_wrong = Signer(b"wrong-key", salt="test-salt").get_signature(value)
    assert signer.verify_signature(value, sig_wrong) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True


# LLM-generated content at query #9
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation (old key)
    old_key = b"old-secret-key"
    new_key = b"new-secret-key"
    signer = Signer([old_key, new_key], salt="test-salt")
    value = b"test-value"
    old_sig = Signer(old_key, salt="test-salt").get_signature(value)
    new_sig = Signer(new_key, salt="test-salt").get_signature(value)
    assert signer.verify_signature(value, old_sig) is True
    assert signer.verify_signature(value, new_sig) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True


# LLM-generated content at query #10
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong-signature")) is False

    # Test with multiple secret keys (key rotation)
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    sig_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with different digest methods
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #11
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with empty signature
    assert signer.verify_signature(value, b"") is False

    # Test with base64 encoded invalid signature
    assert signer.verify_signature(value, base64_encode(b"invalid")) is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = Signer("old-key").get_signature(value)
    signature_new = Signer("new-key").get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old) is True
    assert signer_rotated.verify_signature(value, signature_new) is True
    assert signer_rotated.verify_signature(value, b"invalid") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True
        assert signer.verify_signature(value, b"invalid") is False

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    signature = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature) is True
    assert signer_sha256.verify_signature(value, b"invalid") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True
    assert signer_none.verify_signature(value, b"invalid") is False


# LLM-generated content at query #12
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong-signature")) is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["new-key"]
    assert signer_rotated.verify_signature(value, signature_old) is False

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    value = b"test-value"
    signature = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True


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
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    sig_old = Signer("old-key", salt="test-salt").get_signature(value)
    sig_new = Signer("new-key", salt="test-salt").get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True
    assert signer_rotated.verify_signature(value, b"invalid-sig") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True


# LLM-generated content at query #14
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with string value
    assert signer.verify_signature("test-value", signature) is True

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"rotated-value"
    signature_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["new-key"]  # Simulate key rotation
    assert signer_rotated.verify_signature(value, signature_old) is False

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    value = b"hmac-value"
    signature_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, signature_hmac) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"none-value"
    signature_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature_none) is True


# LLM-generated content at query #15
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation (old key)
    old_key = b"old-secret-key"
    new_key = b"new-secret-key"
    signer_rotated = Signer([old_key, new_key], salt="test-salt")
    value = b"test-value"
    signature_old = Signer(old_key, salt="test-salt").get_signature(value)
    signature_new = Signer(new_key, salt="test-salt").get_signature(value)

    # Old key should still verify
    assert signer_rotated.verify_signature(value, signature_old) is True
    # New key should verify
    assert signer_rotated.verify_signature(value, signature_new) is True
    # Invalid signature should not verify
    assert signer_rotated.verify_signature(value, b"invalid-signature") is False

    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, signature) is True
    assert none_signer.verify_signature(value, b"anything-else") is False

    # Test with HMACAlgorithm and different digest_method
    sha256_signer = Signer(
        "secret-key",
        salt="test-salt",
        digest_method=hashlib.sha256
    )
    value = b"test-value"
    signature = sha256_signer.get_signature(value)
    assert sha256_signer.verify_signature(value, signature) is True
    assert sha256_signer.verify_signature(value, b"invalid-signature") is False

    # Test with malformed base64 signature
    assert signer.verify_signature(value, b"malformed-base64!") is False


# LLM-generated content at query #16
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"invalid-sig")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    sig_old = Signer("old-key", salt="test-salt").get_signature(value)
    sig_new = Signer("new-key", salt="test-salt").get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True
    assert signer_rotated.verify_signature(value, b"invalid-sig") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True
        assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with NoneAlgorithm
    signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    assert signer.verify_signature(value, b"invalid-sig") is False


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Signer_unsign():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)
    assert signer.unsign(signed_value) == value

    # Test with invalid signature
    signer = Signer("secret-key")
    invalid_signed_value = b"test_value.invalid_signature"
    try:
        signer.unsign(invalid_signed_value)
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

    # Test with no separator
    signer = Signer("secret-key")
    no_sep_value = b"test_value"
    try:
        signer.unsign(no_sep_value)
        assert False, "Expected BadSignature exception"
    except BadSignature as e:
        assert "No b'.' found in value" in str(e)

    # Test with multiple separators
    signer = Signer("secret-key")
    value = b"test_value"
    signed_value = signer.sign(value)
    multi_sep_value = signed_value + b".extra"
    try:
        signer.unsign(multi_sep_value)
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

    # Test with key rotation
    signer = Signer(["old-key", "new-key"])
    value = b"test_value"
    signed_value = signer.sign(value)
    assert signer.unsign(signed_value) == value

    # Test with old key
    signer_old = Signer("old-key")
    value = b"test_value"
    signed_value_old = signer_old.sign(value)
    signer_new = Signer(["old-key", "new-key"])
    assert signer_new.unsign(signed_value_old) == value


# LLM-generated content at query #2
#--------------------------

```python
def test_Signer_unsign():
    # Test successful unsigning
    signer = Signer("secret-key")
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with string value
    signer = Signer("secret-key")
    value = "test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == b"test-value"

    # Test with wrong signature
    signer = Signer("secret-key")
    value = b"test-value"
    signed = signer.sign(value)
    wrong_signed = signed[:-1] + b"x"
    try:
        signer.unsign(wrong_signed)
        assert False, "Expected BadSignature"
    except BadSignature:
        pass

    # Test with missing separator
    signer = Signer("secret-key")
    try:
        signer.unsign(b"test-value")
        assert False, "Expected BadSignature"
    except BadSignature as e:
        assert "No b'.' found in value" in str(e)

    # Test with key rotation
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with old key
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    old_signer = Signer("old-key")
    signed = old_signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with NoneAlgorithm
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with different key derivation
    signer = Signer("secret-key", key_derivation="hmac")
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with different digest method
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value


# LLM-generated content at query #3
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret-key")
    derived_key = signer.derive_key()
    assert isinstance(derived_key, bytes)
    assert len(derived_key) > 0

    # Test with concat key derivation
    signer_concat = Signer("secret-key", key_derivation="concat")
    derived_key_concat = signer_concat.derive_key()
    assert isinstance(derived_key_concat, bytes)
    assert len(derived_key_concat) > 0

    # Test with hmac key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    derived_key_hmac = signer_hmac.derive_key()
    assert isinstance(derived_key_hmac, bytes)
    assert len(derived_key_hmac) > 0

    # Test with none key derivation
    signer_none = Signer("secret-key", key_derivation="none")
    derived_key_none = signer_none.derive_key()
    assert derived_key_none == b"secret-key"

    # Test with specific secret_key parameter
    derived_key_specific = signer.derive_key("another-secret-key")
    assert isinstance(derived_key_specific, bytes)
    assert len(derived_key_specific) > 0

    # Test that different key derivations produce different keys
    assert derived_key != derived_key_concat
    assert derived_key != derived_key_hmac
    assert derived_key != derived_key_none
    assert derived_key_concat != derived_key_hmac
    assert derived_key_concat != derived_key_none
    assert derived_key_hmac != derived_key_none

    # Test that same key derivation with same secret produces same key
    signer_same = Signer("secret-key", key_derivation="django-concat")
    assert signer_same.derive_key() == derived_key

    # Test with bytes secret key
    signer_bytes = Signer(b"secret-key")
    derived_key_bytes = signer_bytes.derive_key()
    assert isinstance(derived_key_bytes, bytes)
    assert len(derived_key_bytes) > 0

    # Test with different salt
    signer_salt = Signer("secret-key", salt="different-salt")
    derived_key_salt = signer_salt.derive_key()
    assert derived_key_salt != derived_key

    # Test with different digest method
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    derived_key_sha256 = signer_sha256.derive_key()
    assert isinstance(derived_key_sha256, bytes)
    assert len(derived_key_sha256) > 0
    assert derived_key_sha256 != derived_key

    # Test with invalid key derivation method
    signer_invalid = Signer("secret-key", key_derivation="invalid")
    try:
        signer_invalid.derive_key()
        assert False, "Expected TypeError for invalid key derivation"
    except TypeError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, "not-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    sig_old = Signer("old-key").get_signature(value)
    sig_new = Signer("new-key").get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True
    assert signer_rotated.verify_signature(value, b"wrong-signature") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True
        assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with NoneAlgorithm
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    assert signer.verify_signature(value, b"wrong-signature") is False


# LLM-generated content at query #5
#--------------------------

```python
def test_Signer_unsign():
    # Test successful unsigning
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with string value
    signed_str = signer.sign("test-value")
    assert signer.unsign(signed_str) == b"test-value"

    # Test with custom separator
    signer_custom_sep = Signer("secret-key", salt="test-salt", sep="|")
    signed_custom = signer_custom_sep.sign(value)
    assert signer_custom_sep.unsign(signed_custom) == value

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    signed_old = Signer("old-key", salt="test-salt").sign(value)
    signed_new = Signer("new-key", salt="test-salt").sign(value)
    assert signer_rotated.unsign(signed_old) == value
    assert signer_rotated.unsign(signed_new) == value

    # Test with different key derivation
    signer_hmac = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    signed_hmac = signer_hmac.sign(value)
    assert signer_hmac.unsign(signed_hmac) == value

    # Test with different digest method
    signer_sha256 = Signer("secret-key", salt="test-salt", digest_method=hashlib.sha256)
    signed_sha256 = signer_sha256.sign(value)
    assert signer_sha256.unsign(signed_sha256) == value

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    signed_none = signer_none.sign(value)
    assert signer_none.unsign(signed_none) == value

    # Test invalid signature
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid-signature")

    # Test tampered signature
    signed_tampered = signed[:-1] + b"x"
    with pytest.raises(BadSignature):
        signer.unsign(signed_tampered)

    # Test missing separator
    with pytest.raises(BadSignature):
        signer.unsign(value)

    # Test with wrong key
    wrong_signer = Signer("wrong-key", salt="test-salt")
    with pytest.raises(BadSignature):
        wrong_signer.unsign(signed)


# LLM-generated content at query #6
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"wrong-signature")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    sig_old = Signer("old-key").get_signature(value)
    sig_new = Signer("new-key").get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True
    assert signer_rotated.verify_signature(value, b"wrong-signature") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True
        assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    sig = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig) is True
    assert signer_sha256.verify_signature(value, b"wrong-signature") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig) is True
    assert signer_none.verify_signature(value, b"wrong-signature") is False


# LLM-generated content at query #7
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret", salt="test")
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong_signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong_sig")) is False

    # Test with key rotation
    signer_rotated = Signer(["old_key", "new_key"], salt="test")
    value = b"rotated_value"
    sig_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["newer_key"]
    assert signer_rotated.verify_signature(value, sig_old) is False

    # Test with different key derivation methods
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret", salt="test", key_derivation=derivation)
        value = b"derivation_test"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with custom algorithm
    signer = Signer("secret", salt="test", algorithm=NoneAlgorithm())
    value = b"custom_alg"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True


# LLM-generated content at query #8
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature)

    # Test with incorrect signature
    assert not signer.verify_signature(value, b"wrong-signature")

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"some-signature")
    assert not signer.verify_signature(value, base64_sig)

    # Test with invalid base64 signature
    assert not signer.verify_signature(value, b"invalid-base64!")

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = Signer("old-key").get_signature(value)
    signature_new = Signer("new-key").get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old)
    assert signer_rotated.verify_signature(value, signature_new)

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature)

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    signature = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature)

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature)


# LLM-generated content at query #9
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"some-signature")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"rotated-value"
    signature = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, signature) is True

    # Test with old key in rotation
    old_signer = Signer("old-key")
    old_signature = old_signer.get_signature(value)
    assert signer_rotated.verify_signature(value, old_signature) is True

    # Test with non-matching old key signature
    other_old_signer = Signer("other-old-key")
    other_old_signature = other_old_signer.get_signature(value)
    assert signer_rotated.verify_signature(value, other_old_signature) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"derivation-test"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"sha256-test"
    signature = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"none-algo-test"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True


# LLM-generated content at query #10
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"invalid")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    sig_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["new-key"]
    assert signer_rotated.verify_signature(value, sig_old) is False

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    value = b"test-value"
    sig = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig) is True


# LLM-generated content at query #11
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"wrong-signature")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with multiple secret keys (key rotation)
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    sig_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    sig = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #12
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    sig_old = Signer("old-key").get_signature(value)
    sig_new = Signer("new-key").get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True
    assert signer_rotated.verify_signature(value, b"wrong-signature") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #13
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid_signature") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid_base64!") is False

    # Test with key rotation (old key)
    old_key = b"old-secret-key"
    new_key = b"new-secret-key"
    signer = Signer([old_key, new_key])
    value = b"test_value"
    old_signer = Signer(old_key)
    old_signature = old_signer.get_signature(value)
    assert signer.verify_signature(value, old_signature) is True

    # Test with key rotation (new key)
    new_signer = Signer(new_key)
    new_signature = new_signer.get_signature(value)
    assert signer.verify_signature(value, new_signature) is True

    # Test with key rotation (invalid key)
    invalid_key = b"invalid-secret-key"
    invalid_signer = Signer(invalid_key)
    invalid_signature = invalid_signer.get_signature(value)
    assert signer.verify_signature(value, invalid_signature) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test_value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with custom algorithm (NoneAlgorithm)
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True


# LLM-generated content at query #14
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"not-base64!") is False

    # Test with key rotation (old key)
    old_key = b"old-secret-key"
    new_key = b"new-secret-key"
    signer_rotated = Signer([old_key, new_key], salt="test-salt")
    value = b"test-value"
    signature_old = Signer(old_key, salt="test-salt").get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old) is True

    # Test with key rotation (new key)
    signature_new = Signer(new_key, salt="test-salt").get_signature(value)
    assert signer_rotated.verify_signature(value, signature_new) is True

    # Test with key rotation (invalid key)
    invalid_key = b"invalid-key"
    signature_invalid = Signer(invalid_key, salt="test-salt").get_signature(value)
    assert signer_rotated.verify_signature(value, signature_invalid) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with different digest methods
    for digest_method in [hashlib.sha256, hashlib.sha512]:
        signer = Signer("secret-key", salt="test-salt", digest_method=digest_method)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True


# LLM-generated content at query #15
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation (old key)
    old_key = "old-secret-key"
    new_key = "new-secret-key"
    signer = Signer([old_key, new_key])
    value = b"test-value"
    sig_old = Signer(old_key).get_signature(value)
    sig_new = Signer(new_key).get_signature(value)
    assert signer.verify_signature(value, sig_old) is True
    assert signer.verify_signature(value, sig_new) is True

    # Test with key rotation (invalid key)
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with different digest methods
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True


# LLM-generated content at query #16
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

    # Test with base64 decoding error
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    sig_old = signer_rotated.get_signature(value, secret_key=b"old-key")
    sig_new = signer_rotated.get_signature(value, secret_key=b"new-key")
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True

    # Test with different key derivation methods
    signer_concat = Signer("secret-key", salt="test-salt", key_derivation="concat")
    sig_concat = signer_concat.get_signature(value)
    assert signer_concat.verify_signature(value, sig_concat) is True

    signer_hmac = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True


# LLM-generated content at query #17
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with different value
    assert signer.verify_signature(b"different-value", sig) is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    sig_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["new-key"]
    assert signer_rotated.verify_signature(value, sig_old) is False
    signer_rotated.secret_keys = ["old-key", "new-key"]
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation
    signer_hmac = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac) is True
    assert signer.verify_signature(value, sig_hmac) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True


# LLM-generated content at query #18
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"test-sig")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac) is True
    assert signer.verify_signature(value, sig_hmac) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True
    assert signer.verify_signature(value, sig_none) is False


# LLM-generated content at query #19
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong_signature") is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"test_sig")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    new_sig = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, new_sig) is True

    # Test with old key in rotation
    old_signer = Signer("old-key")
    old_sig = old_signer.get_signature(value)
    assert signer_rotated.verify_signature(value, old_sig) is True

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    hmac_sig = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, hmac_sig) is True

    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid_base64!") is False


# LLM-generated content at query #20
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

    # Test with multiple secret keys (key rotation)
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value_rotated = b"test-value-rotated"
    sig_rotated = signer_rotated.get_signature(value_rotated)
    assert signer_rotated.verify_signature(value_rotated, sig_rotated) is True

    # Test with old key (should still verify)
    old_signer = Signer("old-key", salt="test-salt")
    old_sig = old_signer.get_signature(value_rotated)
    assert signer_rotated.verify_signature(value_rotated, old_sig) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) is True

    # Test with HMACAlgorithm and different digest_method
    custom_signer = Signer(
        "secret-key",
        salt="test-salt",
        algorithm=HMACAlgorithm(digest_method=hashlib.sha256)
    )
    custom_sig = custom_signer.get_signature(value)
    assert custom_signer.verify_signature(value, custom_sig) is True


# LLM-generated content at query #21
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["new-key"]  # Simulate key rotation
    assert signer_rotated.verify_signature(value, signature_old) is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"not-base64!") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True


# LLM-generated content at query #22
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with rotated keys
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = Signer("old-key").get_signature(value)
    signature_new = Signer("new-key").get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old) is True
    assert signer_rotated.verify_signature(value, signature_new) is True
    assert signer_rotated.verify_signature(value, b"wrong-signature") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True
        assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    signature = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature) is True
    assert signer_sha256.verify_signature(value, b"wrong-signature") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True
    assert signer_none.verify_signature(value, b"wrong-signature") is False


# LLM-generated content at query #23
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"test")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["newer-key"]
    assert signer_rotated.verify_signature(value, signature_old) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with custom algorithm
    class CustomAlgorithm(SigningAlgorithm):
        def get_signature(self, key: bytes, value: bytes) -> bytes:
            return b"custom-sig"

    signer = Signer("secret-key", algorithm=CustomAlgorithm())
    value = b"test-value"
    assert signer.verify_signature(value, b"custom-sig") is True
    assert signer.verify_signature(value, b"wrong-sig") is False


# LLM-generated content at query #24
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old) is True

    # Test with base64 decode failure
    assert signer.verify_signature(value, b"not-base64!") is False

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    value = b"test-value"
    signature_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, signature_hmac) is True
    assert signer.verify_signature(value, signature_hmac) is False

    # Test with different digest method
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    signature_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature_sha256) is True
    assert signer.verify_signature(value, signature_sha256) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature_none) is True
    assert signer.verify_signature(value, signature_none) is False


# LLM-generated content at query #25
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    wrong_sig = b"wrong-signature"
    assert signer.verify_signature(value, wrong_sig) is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"test-sig")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    invalid_base64_sig = b"invalid-base64"
    assert signer.verify_signature(value, invalid_base64_sig) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    sig_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    sig = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig) is True


# LLM-generated content at query #26
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with base64 decode error
    assert signer.verify_signature(value, "not-base64") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)  # Uses new-key
    # Manually create signature with old-key for testing
    old_key_signer = Signer("old-key", salt="test-salt")
    signature_old_key = old_key_signer.get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old_key) is True
    assert signer_rotated.verify_signature(value, signature_old) is True

    # Test with different key derivation
    signer_hmac = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    signature_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, signature_hmac) is True
    assert signer.verify_signature(value, signature_hmac) is False  # Different key derivation


# LLM-generated content at query #27
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    # Change the key to simulate rotation
    signer_rotated.secret_keys = ["new-key"]
    assert signer_rotated.verify_signature(value, signature_old) is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"not-base64!") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True
    assert signer_none.verify_signature(value, b"") is True  # Empty signature is valid for NoneAlgorithm

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True


# LLM-generated content at query #28
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong-signature")) is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["new-key"]
    assert signer_rotated.verify_signature(value, signature_old) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True
    assert signer_none.verify_signature(value, b"") is True  # Empty signature is valid for NoneAlgorithm


# LLM-generated content at query #29
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong_signature") is False

    # Test with base64 encoded signature
    sig_b64 = base64_encode(b"wrong_sig")
    assert signer.verify_signature(value, sig_b64) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid_base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old_key", "new_key"])
    value = b"rotated_value"
    sig_old = signer_rotated.get_signature(value)  # uses new_key
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with old key
    old_signer = Signer("old_key", salt=signer_rotated.salt)
    sig_old = old_signer.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with wrong key
    wrong_signer = Signer("wrong_key")
    sig_wrong = wrong_signer.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_wrong) is False

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac) is True
    assert signer.verify_signature(value, sig_hmac) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True
    assert signer.verify_signature(value, sig_none) is False


# LLM-generated content at query #30
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    # Create a new signer with only the new key to simulate key rotation
    signer_new = Signer("new-key")
    signature_new = signer_new.get_signature(value)
    # Old signature should not verify with new key
    assert signer_new.verify_signature(value, signature_old) is False
    # New signature should verify with rotated keys
    assert signer_rotated.verify_signature(value, signature_new) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    signature = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #31
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong-signature")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value, secret_key=b"old-key")
    sig_new = signer_rotated.get_signature(value, secret_key=b"new-key")
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_kd = Signer("secret-key", key_derivation=key_derivation)
        sig_kd = signer_kd.get_signature(value)
        assert signer_kd.verify_signature(value, sig_kd) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    sig_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig_sha256) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #32
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    # Change the key to new-key and verify old signature still works
    assert signer_rotated.verify_signature(value, signature_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #33
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with empty signature
    assert signer.verify_signature(value, b"") is False

    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) is True

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    old_sig = signer_rotated.get_signature(value, secret_key=b"old-key")
    new_sig = signer_rotated.get_signature(value, secret_key=b"new-key")
    assert signer_rotated.verify_signature(value, old_sig) is True
    assert signer_rotated.verify_signature(value, new_sig) is True
    assert signer_rotated.verify_signature(value, b"wrong-signature") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_deriv = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        deriv_sig = signer_deriv.get_signature(value)
        assert signer_deriv.verify_signature(value, deriv_sig) is True


# LLM-generated content at query #34
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation (old key)
    old_key = "old-secret-key"
    new_key = "new-secret-key"
    signer = Signer([old_key, new_key], salt="test-salt")
    value = b"test-value"
    old_sig = Signer(old_key, salt="test-salt").get_signature(value)
    assert signer.verify_signature(value, old_sig) is True

    # Test with key rotation (new key)
    new_sig = Signer(new_key, salt="test-salt").get_signature(value)
    assert signer.verify_signature(value, new_sig) is True

    # Test with key rotation (invalid key)
    invalid_sig = Signer("invalid-key", salt="test-salt").get_signature(value)
    assert signer.verify_signature(value, invalid_sig) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with custom algorithm (NoneAlgorithm)
    signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True


# LLM-generated content at query #35
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    sig_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["new-key"]  # Simulate key rotation
    assert signer_rotated.verify_signature(value, sig_old) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with custom digest method
    signer = Signer("secret-key", salt="test-salt", digest_method=hashlib.sha256)
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True


# LLM-generated content at query #36
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = Signer("old-key").get_signature(value)
    signature_new = Signer("new-key").get_signature(value)

    # Old key should still verify
    assert signer_rotated.verify_signature(value, signature_old) is True
    # New key should verify
    assert signer_rotated.verify_signature(value, signature_new) is True
    # Wrong key should not verify
    assert signer_rotated.verify_signature(value, b"wrong-signature") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True


# LLM-generated content at query #37
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"wrong-signature")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with different digest methods
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with custom algorithm
    class CustomAlgorithm(SigningAlgorithm):
        def get_signature(self, key: bytes, value: bytes) -> bytes:
            return b"custom-signature"

    signer = Signer("secret-key", algorithm=CustomAlgorithm())
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True


# LLM-generated content at query #38
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"wrong-signature")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_kd = Signer("secret-key", key_derivation=key_derivation)
        sig_kd = signer_kd.get_signature(value)
        assert signer_kd.verify_signature(value, sig_kd) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    sig_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig_sha256) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True


# LLM-generated content at query #39
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    sig_old = signer_rotated.get_signature(value, secret_key=b"old-key")
    sig_new = signer_rotated.get_signature(value, secret_key=b"new-key")
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_kd = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        sig_kd = signer_kd.get_signature(value)
        assert signer_kd.verify_signature(value, sig_kd) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True


# LLM-generated content at query #40
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    signature_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old) is True

    # Test with base64 decode failure
    assert signer.verify_signature(value, b"not-base64!") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    assert signer_none.verify_signature(value, b"") is True
    assert signer_none.verify_signature(value, b"anything") is False


# LLM-generated content at query #41
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["newer-key"]
    assert signer_rotated.verify_signature(value, signature_old) is False

    # Test with base64 decode error
    assert signer.verify_signature(value, "invalid-base64") is False

    # Test with different key derivation
    signer_hmac = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    signature_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, signature_hmac) is True
    assert signer.verify_signature(value, signature_hmac) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    signature_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature_none) is True


# LLM-generated content at query #42
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong-signature")) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_kd = Signer("secret-key", key_derivation=key_derivation)
        sig_kd = signer_kd.get_signature(value)
        assert signer_kd.verify_signature(value, sig_kd) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    sig_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig_sha256) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True


# LLM-generated content at query #43
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with different value
    assert signer.verify_signature(b"different-value", sig) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value, secret_key="old-key")
    sig_new = signer_rotated.get_signature(value, secret_key="new-key")
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True
    assert signer_rotated.verify_signature(value, b"invalid-sig") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True
    assert signer_none.verify_signature(value, b"invalid-sig") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #44
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"not-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_deriv = Signer("secret-key", key_derivation=key_derivation)
        sig_deriv = signer_deriv.get_signature(value)
        assert signer_deriv.verify_signature(value, sig_deriv) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    sig_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig_sha256) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True


# LLM-generated content at query #45
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with empty signature
    assert signer.verify_signature(value, b"") is False

    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    none_signature = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_signature) is True

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value_rotated = b"rotated-value"
    signature_rotated = signer_rotated.get_signature(value_rotated)
    assert signer_rotated.verify_signature(value_rotated, signature_rotated) is True

    # Test with old key (should still verify)
    old_signer = Signer("old-key", salt="test-salt")
    old_signature = old_signer.get_signature(value_rotated)
    assert signer_rotated.verify_signature(value_rotated, old_signature) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #46
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    wrong_sig = b"wrong-signature"
    assert signer.verify_signature(value, wrong_sig) is False

    # Test with invalid base64 signature
    invalid_sig = b"invalid-base64"
    assert signer.verify_signature(value, invalid_sig) is False

    # Test with key rotation (old key)
    old_key = b"old-secret-key"
    new_key = b"new-secret-key"
    signer = Signer([old_key, new_key], salt="test-salt")
    value = b"test-value"
    sig_old = Signer(old_key, salt="test-salt").get_signature(value)
    sig_new = Signer(new_key, salt="test-salt").get_signature(value)
    assert signer.verify_signature(value, sig_old) is True
    assert signer.verify_signature(value, sig_new) is True

    # Test with wrong key
    wrong_key_signer = Signer("wrong-secret-key", salt="test-salt")
    wrong_sig = wrong_key_signer.get_signature(value)
    assert signer.verify_signature(value, wrong_sig) is False

    # Test with different key derivation
    signer_hmac = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac) is True
    assert signer.verify_signature(value, sig_hmac) is False

    # Test with different digest method
    signer_sha256 = Signer("secret-key", salt="test-salt", digest_method=hashlib.sha256)
    sig_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig_sha256) is True
    assert signer.verify_signature(value, sig_sha256) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True
    assert signer.verify_signature(value, sig_none) is False


# LLM-generated content at query #47
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    sig_old = Signer("old-key", salt="test-salt").get_signature(value)
    sig_new = Signer("new-key", salt="test-salt").get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True
    assert signer_rotated.verify_signature(value, b"wrong-sig") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True
        assert signer.verify_signature(value, b"wrong-sig") is False

    # Test with NoneAlgorithm
    signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True
    assert signer.verify_signature(value, b"wrong-sig") is False


# LLM-generated content at query #48
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with base64 decode error
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with rotated keys (old key)
    old_key = b"old-secret-key"
    new_key = b"new-secret-key"
    signer = Signer([old_key, new_key])
    value = b"test-value"
    old_sig = Signer(old_key).get_signature(value)
    new_sig = Signer(new_key).get_signature(value)
    assert signer.verify_signature(value, old_sig) is True
    assert signer.verify_signature(value, new_sig) is True

    # Test with rotated keys (invalid key)
    invalid_key = b"invalid-secret-key"
    invalid_sig = Signer(invalid_key).get_signature(value)
    assert signer.verify_signature(value, invalid_sig) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True


