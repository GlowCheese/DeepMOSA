####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Signer_unsign():
    # Test successful unsigning
    signer = Signer("secret-key")
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with string value
    assert signer.unsign(signed.decode()) == value

    # Test with wrong signature
    wrong_signed = signed[:-1] + b"x"
    try:
        signer.unsign(wrong_signed)
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test with no separator
    try:
        signer.unsign(b"no-separator")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test with multiple separators
    multi_sep = b"value.sep1.sep2"
    try:
        signer.unsign(multi_sep)
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    signed_old = Signer("old-key").sign(value)
    signed_new = Signer("new-key").sign(value)
    assert signer_rotated.unsign(signed_old) == value
    assert signer_rotated.unsign(signed_new) == value

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    signed_hmac = signer_hmac.sign(value)
    assert signer_hmac.unsign(signed_hmac) == value

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    signed_none = signer_none.sign(value)
    assert signer_none.unsign(signed_none) == value


# LLM-generated content at query #2
#--------------------------

```python
def test_Signer_unsign():
    # Test with valid signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with invalid signature
    signer = Signer("secret-key", salt="test-salt")
    with pytest.raises(BadSignature):
        signer.unsign(b"test-value.invalid-sig")

    # Test with no separator
    signer = Signer("secret-key", salt="test-salt")
    with pytest.raises(BadSignature):
        signer.unsign(b"test-value")

    # Test with multiple separators
    signer = Signer("secret-key", salt="test-salt")
    value = b"test.value.with.separators"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with key rotation
    signer = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with old key
    signer = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    old_signer = Signer("old-key", salt="test-salt")
    signed = old_signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with different key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with different digest method
    signer = Signer("secret-key", salt="test-salt", digest_method=hashlib.sha256)
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

    # Test with NoneAlgorithm
    signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value


# LLM-generated content at query #3
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with concat key derivation
    signer_concat = Signer("secret-key", salt="test-salt", key_derivation="concat")
    derived_key_concat = signer_concat.derive_key()
    expected_key_concat = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert derived_key_concat == expected_key_concat

    # Test with hmac key derivation
    signer_hmac = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    derived_key_hmac = signer_hmac.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key_hmac = mac.digest()
    assert derived_key_hmac == expected_key_hmac

    # Test with none key derivation
    signer_none = Signer("secret-key", salt="test-salt", key_derivation="none")
    derived_key_none = signer_none.derive_key()
    assert derived_key_none == b"secret-key"

    # Test with specific secret_key parameter
    derived_key_specific = signer.derive_key("another-secret")
    expected_key_specific = hashlib.sha1(b"test-salt" + b"signer" + b"another-secret").digest()
    assert derived_key_specific == expected_key_specific

    # Test with bytes secret key
    signer_bytes = Signer(b"secret-key", salt=b"test-salt")
    derived_key_bytes = signer_bytes.derive_key()
    expected_key_bytes = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key_bytes == expected_key_bytes

    # Test with key rotation (multiple secret keys)
    signer_rotation = Signer(["old-key", "new-key"], salt="test-salt")
    derived_key_rotation = signer_rotation.derive_key()
    expected_key_rotation = hashlib.sha1(b"test-salt" + b"signer" + b"new-key").digest()
    assert derived_key_rotation == expected_key_rotation

    # Test with key rotation and specific secret_key
    derived_key_rotation_specific = signer_rotation.derive_key("old-key")
    expected_key_rotation_specific = hashlib.sha1(b"test-salt" + b"signer" + b"old-key").digest()
    assert derived_key_rotation_specific == expected_key_rotation_specific

    # Test with custom digest method
    def custom_digest(string=b""):
        return hashlib.sha256(string)

    signer_custom = Signer("secret-key", salt="test-salt", digest_method=custom_digest)
    derived_key_custom = signer_custom.derive_key()
    expected_key_custom = hashlib.sha256(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key_custom == expected_key_custom


# LLM-generated content at query #4
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
    signature_old = Signer("old-key").get_signature(value)
    signature_new = Signer("new-key").get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old) is True
    assert signer_rotated.verify_signature(value, signature_new) is True
    assert signer_rotated.verify_signature(value, b"wrong-signature") is False

    # Test with different key derivation
    signer_concat = Signer("secret-key", key_derivation="concat")
    signature_concat = signer_concat.get_signature(value)
    assert signer_concat.verify_signature(value, signature_concat) is True
    assert signer.verify_signature(value, signature_concat) is False

    # Test with different digest method
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    signature_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature_sha256) is True
    assert signer.verify_signature(value, signature_sha256) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    signature_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature_none) is True
    assert signer.verify_signature(value, signature_none) is False


# LLM-generated content at query #5
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key_derivation ('django-concat')
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key()
    expected_key = _lazy_sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with 'concat' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    derived_key = signer.derive_key()
    expected_key = _lazy_sha1(b"test-salt" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with 'hmac' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    derived_key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=_lazy_sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert derived_key == expected_key

    # Test with 'none' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    derived_key = signer.derive_key()
    assert derived_key == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key("another-secret")
    expected_key = _lazy_sha1(b"test-salt" + b"signer" + b"another-secret").digest()
    assert derived_key == expected_key

    # Test with bytes secret_key
    signer = Signer(b"secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = _lazy_sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with bytes salt
    signer = Signer("secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = _lazy_sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with invalid key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="invalid")
    try:
        signer.derive_key()
        assert False, "Expected TypeError for invalid key_derivation"
    except TypeError:
        pass


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

    # Test with base64 decode error
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
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret-key")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"itsdangerous.Signersignersecret-key").digest()
    assert derived_key == expected_key

    # Test with concat key derivation
    signer = Signer("secret-key", key_derivation="concat")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"itsdangerous.Signersecret-key").digest()
    assert derived_key == expected_key

    # Test with hmac key derivation
    signer = Signer("secret-key", key_derivation="hmac")
    derived_key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"itsdangerous.Signer")
    expected_key = mac.digest()
    assert derived_key == expected_key

    # Test with none key derivation
    signer = Signer("secret-key", key_derivation="none")
    derived_key = signer.derive_key()
    assert derived_key == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key")
    derived_key = signer.derive_key("other-secret")
    expected_key = hashlib.sha1(b"itsdangerous.Signersignerother-secret").digest()
    assert derived_key == expected_key

    # Test with bytes secret_key
    signer = Signer(b"secret-key")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"itsdangerous.Signersignersecret-key").digest()
    assert derived_key == expected_key

    # Test with bytes secret_key parameter
    signer = Signer("secret-key")
    derived_key = signer.derive_key(b"other-secret")
    expected_key = hashlib.sha1(b"itsdangerous.Signersignerother-secret").digest()
    assert derived_key == expected_key

    # Test with invalid key derivation method
    signer = Signer("secret-key", key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer.derive_key()


# LLM-generated content at query #8
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
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

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

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #9
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key_derivation ('django-concat')
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) > 0

    # Test with 'concat' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert key == expected_key

    # Test with 'django-concat' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="django-concat")
    key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert key == expected_key

    # Test with 'hmac' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert key == expected_key

    # Test with 'none' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key("specific-key")
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"specific-key").digest()
    assert key == expected_key

    # Test with bytes secret_key
    signer = Signer(b"secret-key", salt=b"test-salt")
    key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert key == expected_key

    # Test with unknown key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="unknown")
    with pytest.raises(TypeError):
        signer.derive_key()


# LLM-generated content at query #10
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key_derivation (django-concat)
    signer = Signer("secret-key", salt="test-salt")
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert signer.derive_key() == expected_key

    # Test with concat key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    expected_key = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert signer.derive_key() == expected_key

    # Test with hmac key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert signer.derive_key() == expected_key

    # Test with none key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    assert signer.derive_key() == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    assert signer.derive_key("other-key") == hashlib.sha1(b"test-salt" + b"signer" + b"other-key").digest()

    # Test with bytes secret_key
    signer = Signer(b"secret-key", salt=b"test-salt")
    assert signer.derive_key() == hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()

    # Test with bytes salt
    signer = Signer("secret-key", salt=b"test-salt")
    assert signer.derive_key() == hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()

    # Test with unknown key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="unknown")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer.derive_key()


# LLM-generated content at query #11
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key_derivation ("django-concat")
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) > 0

    # Test with "concat" key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert key == expected_key

    # Test with "hmac" key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert key == expected_key

    # Test with "none" key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret-key"

    # Test with custom secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key("custom-secret")
    mac = hmac.new(b"custom-secret", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert key == expected_key

    # Test with invalid key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer.derive_key()


# LLM-generated content at query #12
#--------------------------

```python
def test_Signer_derive_key():
    # Test default key derivation (django-concat)
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test concat key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test hmac key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    derived_key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert derived_key == expected_key

    # Test none key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    derived_key = signer.derive_key()
    assert derived_key == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key("other-key")
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"other-key").digest()
    assert derived_key == expected_key

    # Test with bytes secret key
    signer = Signer(b"secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with bytes salt
    signer = Signer("secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with invalid key derivation method
    signer = Signer("secret-key", salt="test-salt", key_derivation="invalid")
    with pytest.raises(TypeError):
        signer.derive_key()


# LLM-generated content at query #13
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
    assert signer.verify_signature(value, base64_encode(b"invalid-sig")) is False

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


# LLM-generated content at query #14
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key_derivation (django-concat)
    signer = Signer("secret", salt="test_salt")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) > 0

    # Test with concat key_derivation
    signer = Signer("secret", salt="test_salt", key_derivation="concat")
    key = signer.derive_key()
    expected_key = hashlib.sha1(b"test_saltsecret").digest()
    assert key == expected_key

    # Test with django-concat key_derivation
    signer = Signer("secret", salt="test_salt", key_derivation="django-concat")
    key = signer.derive_key()
    expected_key = hashlib.sha1(b"test_saltsignersecret").digest()
    assert key == expected_key

    # Test with hmac key_derivation
    signer = Signer("secret", salt="test_salt", key_derivation="hmac")
    key = signer.derive_key()
    mac = hmac.new(b"secret", digestmod=hashlib.sha1)
    mac.update(b"test_salt")
    expected_key = mac.digest()
    assert key == expected_key

    # Test with none key_derivation
    signer = Signer("secret", salt="test_salt", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret"

    # Test with specific secret_key parameter
    signer = Signer("secret", salt="test_salt", key_derivation="concat")
    key = signer.derive_key("specific_secret")
    expected_key = hashlib.sha1(b"test_salt" + b"specific_secret").digest()
    assert key == expected_key

    # Test with bytes secret_key
    signer = Signer(b"secret", salt=b"test_salt", key_derivation="concat")
    key = signer.derive_key()
    expected_key = hashlib.sha1(b"test_salt" + b"secret").digest()
    assert key == expected_key

    # Test with different digest_method
    signer = Signer("secret", salt="test_salt", key_derivation="concat", digest_method=hashlib.sha256)
    key = signer.derive_key()
    expected_key = hashlib.sha256(b"test_saltsecret").digest()
    assert key == expected_key

    # Test with invalid key_derivation
    signer = Signer("secret", salt="test_salt", key_derivation="invalid")
    try:
        signer.derive_key()
        assert False, "Expected TypeError for invalid key_derivation"
    except TypeError:
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret", salt="test_salt")
    key = signer.derive_key()
    expected = hashlib.sha1(b"test_salt" + b"signer" + b"secret").digest()
    assert key == expected

    # Test with concat key derivation
    signer = Signer("secret", salt="test_salt", key_derivation="concat")
    key = signer.derive_key()
    expected = hashlib.sha1(b"test_salt" + b"secret").digest()
    assert key == expected

    # Test with hmac key derivation
    signer = Signer("secret", salt="test_salt", key_derivation="hmac")
    key = signer.derive_key()
    expected = hmac.new(b"secret", digestmod=hashlib.sha1).update(b"test_salt").digest()
    assert key == expected

    # Test with none key derivation
    signer = Signer("secret", salt="test_salt", key_derivation="none")
    key = signer.derive_key()
    expected = b"secret"
    assert key == expected

    # Test with specific secret_key parameter
    signer = Signer("secret", salt="test_salt")
    key = signer.derive_key("other_secret")
    expected = hashlib.sha1(b"test_salt" + b"signer" + b"other_secret").digest()
    assert key == expected

    # Test with bytes secret_key
    signer = Signer(b"secret", salt="test_salt")
    key = signer.derive_key()
    expected = hashlib.sha1(b"test_salt" + b"signer" + b"secret").digest()
    assert key == expected

    # Test with bytes salt
    signer = Signer("secret", salt=b"test_salt")
    key = signer.derive_key()
    expected = hashlib.sha1(b"test_salt" + b"signer" + b"secret").digest()
    assert key == expected

    # Test with different digest_method
    signer = Signer("secret", salt="test_salt", digest_method=hashlib.sha256)
    key = signer.derive_key()
    expected = hashlib.sha256(b"test_salt" + b"signer" + b"secret").digest()
    assert key == expected

    # Test with invalid key_derivation
    signer = Signer("secret", salt="test_salt", key_derivation="invalid")
    with pytest.raises(TypeError):
        signer.derive_key()


# LLM-generated content at query #16
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret", salt="salt")
    key = signer.derive_key()
    assert key == hashlib.sha1(b"salt" + b"signer" + b"secret").digest()

    # Test with concat key derivation
    signer = Signer("secret", salt="salt", key_derivation="concat")
    key = signer.derive_key()
    assert key == hashlib.sha1(b"salt" + b"secret").digest()

    # Test with hmac key derivation
    signer = Signer("secret", salt="salt", key_derivation="hmac")
    key = signer.derive_key()
    mac = hmac.new(b"secret", digestmod=hashlib.sha1)
    mac.update(b"salt")
    assert key == mac.digest()

    # Test with none key derivation
    signer = Signer("secret", salt="salt", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret"

    # Test with specific secret_key parameter
    signer = Signer("secret", salt="salt")
    key = signer.derive_key("other_secret")
    assert key == hashlib.sha1(b"salt" + b"signer" + b"other_secret").digest()

    # Test with bytes secret_key
    signer = Signer(b"secret", salt=b"salt")
    key = signer.derive_key(b"other_secret")
    assert key == hashlib.sha1(b"salt" + b"signer" + b"other_secret").digest()

    # Test with invalid key derivation
    signer = Signer("secret", salt="salt", key_derivation="invalid")
    with pytest.raises(TypeError):
        signer.derive_key()


# LLM-generated content at query #17
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret", salt="salt")
    assert signer.derive_key() == hashlib.sha1(b"salt" + b"signer" + b"secret").digest()

    # Test with concat key derivation
    signer = Signer("secret", salt="salt", key_derivation="concat")
    assert signer.derive_key() == hashlib.sha1(b"salt" + b"secret").digest()

    # Test with hmac key derivation
    signer = Signer("secret", salt="salt", key_derivation="hmac")
    mac = hmac.new(b"secret", digestmod=hashlib.sha1)
    mac.update(b"salt")
    assert signer.derive_key() == mac.digest()

    # Test with none key derivation
    signer = Signer("secret", salt="salt", key_derivation="none")
    assert signer.derive_key() == b"secret"

    # Test with specific secret_key parameter
    signer = Signer("secret", salt="salt")
    assert signer.derive_key("other_secret") == hashlib.sha1(b"salt" + b"signer" + b"other_secret").digest()

    # Test with bytes secret_key
    signer = Signer(b"secret", salt=b"salt")
    assert signer.derive_key() == hashlib.sha1(b"salt" + b"signer" + b"secret").digest()

    # Test with invalid key derivation
    with pytest.raises(TypeError):
        signer = Signer("secret", salt="salt", key_derivation="invalid")
        signer.derive_key()


# LLM-generated content at query #18
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key_derivation ('django-concat')
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with 'concat' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with 'hmac' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    derived_key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert derived_key == expected_key

    # Test with 'none' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    derived_key = signer.derive_key()
    assert derived_key == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key("other-secret")
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"other-secret").digest()
    assert derived_key == expected_key

    # Test with bytes secret_key
    signer = Signer(b"secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with bytes salt
    signer = Signer("secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with invalid key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="invalid")
    with pytest.raises(TypeError):
        signer.derive_key()


# LLM-generated content at query #19
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key_derivation ('django-concat')
    signer = Signer("secret", salt="test-salt")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) > 0

    # Test with 'concat' key_derivation
    signer_concat = Signer("secret", salt="test-salt", key_derivation="concat")
    key_concat = signer_concat.derive_key()
    assert key_concat != key  # Different derivation methods should produce different keys

    # Test with 'hmac' key_derivation
    signer_hmac = Signer("secret", salt="test-salt", key_derivation="hmac")
    key_hmac = signer_hmac.derive_key()
    assert key_hmac != key  # Different derivation methods should produce different keys

    # Test with 'none' key_derivation
    signer_none = Signer("secret", salt="test-salt", key_derivation="none")
    key_none = signer_none.derive_key()
    assert key_none == b"secret"  # 'none' should return the secret key as-is

    # Test with custom secret_key parameter
    custom_key = signer.derive_key("custom-secret")
    assert custom_key != key  # Different secret keys should produce different keys

    # Test with bytes secret_key
    signer_bytes = Signer(b"secret", salt="test-salt")
    key_bytes = signer_bytes.derive_key()
    assert key_bytes == key  # Bytes and str secret keys should produce the same key

    # Test with invalid key_derivation
    signer_invalid = Signer("secret", salt="test-salt", key_derivation="invalid")
    with pytest.raises(TypeError):
        signer_invalid.derive_key()


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

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    sig_old = signer_rotated.get_signature(value)
    # Change the key to new-key and verify old signature still works
    signer_rotated.secret_keys = ["old-key", "new-key"]
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", salt="test-salt", digest_method=hashlib.sha256)
    value = b"test-value"
    sig = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig) is True


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

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"invalid")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    signature_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_kd = Signer("secret-key", key_derivation=key_derivation)
        signature_kd = signer_kd.get_signature(value)
        assert signer_kd.verify_signature(value, signature_kd) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    signature_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature_sha256) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    signature_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature_none) is True


# LLM-generated content at query #22
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


# LLM-generated content at query #23
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
    assert signer.verify_signature(value, b"not-base64!") is False

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
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"invalid-sig")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac) is True

    # Test with different digest method
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    sig_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig_sha256) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True


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
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

    # Test with different value
    assert signer.verify_signature(b"different-value", sig) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with empty signature
    assert signer.verify_signature(value, b"") is False

    # Test with NoneAlgorithm
    none_signer = Signer("secret-key", algorithm=NoneAlgorithm())
    none_sig = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_sig) is True
    assert none_signer.verify_signature(value, b"anything") is True

    # Test with different key derivation
    hmac_signer = Signer("secret-key", key_derivation="hmac")
    hmac_sig = hmac_signer.get_signature(value)
    assert hmac_signer.verify_signature(value, hmac_sig) is True
    assert signer.verify_signature(value, hmac_sig) is False


# LLM-generated content at query #26
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
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value, secret_key=b"old-key")
    sig_new = signer_rotated.get_signature(value, secret_key=b"new-key")
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_deriv = Signer("secret-key", key_derivation=key_derivation)
        sig_deriv = signer_deriv.get_signature(value)
        assert signer_deriv.verify_signature(value, sig_deriv) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


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
    assert signer_rotated.verify_signature(value, signature_old) is True

    # Test with base64 decode error
    assert signer.verify_signature(value, b"not-base64!") is False

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
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    assert signer.verify_signature(value, b"wrong-signature") is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
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


# LLM-generated content at query #29
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

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value, secret_key=b"old-key")
    sig_new = signer_rotated.get_signature(value, secret_key=b"new-key")
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True
    assert signer_rotated.verify_signature(value, b"invalid-sig") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_kd = Signer("secret-key", key_derivation=key_derivation)
        sig_kd = signer_kd.get_signature(value)
        assert signer_kd.verify_signature(value, sig_kd) is True
        assert signer_kd.verify_signature(value, b"invalid-sig") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True
    assert signer_none.verify_signature(value, b"invalid-sig") is False


# LLM-generated content at query #30
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
    base64_sig = base64_encode(b"wrong-sig")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"rotated-value"
    signature_old = Signer("old-key").get_signature(value)
    signature_new = Signer("new-key").get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old) is True
    assert signer_rotated.verify_signature(value, signature_new) is True
    assert signer_rotated.verify_signature(value, b"wrong-sig") is False

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
    assert signer.verify_signature(value, b"wrong-sig") is False

    # Test with base64 encoded signature
    sig_b64 = base64_encode(b"wrong-sig")
    assert signer.verify_signature(value, sig_b64) is False

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
    base64_sig = base64_encode(b"wrong-sig")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"rotated-value"
    signature_old = Signer("old-key").get_signature(value)
    signature_new = Signer("new-key").get_signature(value)

    # Should verify with new key
    assert signer_rotated.verify_signature(value, signature_new) is True
    # Should verify with old key
    assert signer_rotated.verify_signature(value, signature_old) is True
    # Should not verify with unrelated key
    assert signer_rotated.verify_signature(value, Signer("other-key").get_signature(value)) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_kd = Signer("secret-key", key_derivation=key_derivation)
        value = b"kd-value"
        signature = signer_kd.get_signature(value)
        assert signer_kd.verify_signature(value, signature) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"sha256-value"
    signature = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"none-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True


# LLM-generated content at query #33
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


# LLM-generated content at query #34
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
    signer_rotated.secret_keys = ["new-key"]  # Simulate key rotation
    assert signer_rotated.verify_signature(value, signature_old) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True
    assert signer_none.verify_signature(value, b"") is True  # Empty signature is valid for NoneAlgorithm


# LLM-generated content at query #35
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

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with key rotation
    signer = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    signature_old = Signer("old-key", salt="test-salt").get_signature(value)
    signature_new = Signer("new-key", salt="test-salt").get_signature(value)
    assert signer.verify_signature(value, signature_old) is True
    assert signer.verify_signature(value, signature_new) is True
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with NoneAlgorithm
    signer = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True
    assert signer.verify_signature(value, b"invalid-signature") is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #36
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

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"invalid-sig")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    sig_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["new-key"]
    sig_new = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_new) is True
    assert signer_rotated.verify_signature(value, sig_old) is False

    # Test with different key derivation
    signer_concat = Signer("secret-key", key_derivation="concat")
    sig_concat = signer_concat.get_signature(value)
    assert signer_concat.verify_signature(value, sig_concat) is True
    assert signer.verify_signature(value, sig_concat) is False

    # Test with different digest method
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    sig_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig_sha256) is True
    assert signer.verify_signature(value, sig_sha256) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True
    assert signer.verify_signature(value, sig_none) is False


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key_derivation (django-concat)
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with concat key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with hmac key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    derived_key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert derived_key == expected_key

    # Test with none key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    derived_key = signer.derive_key()
    assert derived_key == b"secret-key"

    # Test with custom secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key("custom-key")
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"custom-key").digest()
    assert derived_key == expected_key

    # Test with bytes secret_key
    signer = Signer(b"secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with bytes salt
    signer = Signer("secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with key rotation (multiple secret keys)
    signer = Signer(["old-key", "new-key"], salt="test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"new-key").digest()
    assert derived_key == expected_key

    # Test with key rotation and custom secret_key parameter
    signer = Signer(["old-key", "new-key"], salt="test-salt")
    derived_key = signer.derive_key("old-key")
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"old-key").digest()
    assert derived_key == expected_key

    # Test with unknown key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="unknown")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer.derive_key()


# LLM-generated content at query #2
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with concat key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with hmac key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    derived_key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert derived_key == expected_key

    # Test with none key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    derived_key = signer.derive_key()
    assert derived_key == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key("other-secret")
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"other-secret").digest()
    assert derived_key == expected_key

    # Test with bytes secret_key
    signer = Signer(b"secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with bytes salt
    signer = Signer("secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with invalid key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer.derive_key()


# LLM-generated content at query #3
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

    # Test with wrong value
    assert signer.verify_signature(b"wrong-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = Signer("old-key").get_signature(value)
    signature_new = Signer("new-key").get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old) is True
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

    # Test with base64 encoded signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = base64_encode(b"test-signature")
    assert signer.verify_signature(value, signature) is False


# LLM-generated content at query #4
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert key == signer.digest_method(signer.salt + b"signer" + b"secret-key").digest()

    # Test with concat key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert key == signer.digest_method(b"test-salt" + b"secret-key").digest()

    # Test with hmac key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    mac = hmac.new(b"secret-key", digestmod=signer.digest_method)
    mac.update(b"test-salt")
    assert key == mac.digest()

    # Test with none key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert key == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key("other-secret")
    assert isinstance(key, bytes)
    assert key == signer.digest_method(b"test-salt" + b"signer" + b"other-secret").digest()

    # Test with bytes secret key
    signer = Signer(b"secret-key", salt=b"test-salt")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert key == signer.digest_method(b"test-salt" + b"signer" + b"secret-key").digest()

    # Test with invalid key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer.derive_key()


# LLM-generated content at query #5
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

    # Test with base64 decoding error
    assert signer.verify_signature(value, b"not-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    old_sig = Signer("old-key").get_signature(value)
    new_sig = Signer("new-key").get_signature(value)
    assert signer_rotated.verify_signature(value, old_sig) is True
    assert signer_rotated.verify_signature(value, new_sig) is True
    assert signer_rotated.verify_signature(value, b"wrong-sig") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    signature_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature_sha256) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    signature_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature_none) is True


# LLM-generated content at query #6
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret-key")
    assert signer.derive_key() == signer.digest_method(signer.salt + b"signer" + b"secret-key").digest()

    # Test with concat key derivation
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.derive_key() == signer.digest_method(signer.salt + b"secret-key").digest()

    # Test with hmac key derivation
    signer = Signer("secret-key", key_derivation="hmac")
    mac = hmac.new(b"secret-key", digestmod=signer.digest_method)
    mac.update(signer.salt)
    assert signer.derive_key() == mac.digest()

    # Test with none key derivation
    signer = Signer("secret-key", key_derivation="none")
    assert signer.derive_key() == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", key_derivation="django-concat")
    assert signer.derive_key("another-secret") == signer.digest_method(signer.salt + b"signer" + b"another-secret").digest()

    # Test with bytes secret key
    signer = Signer(b"secret-key", key_derivation="django-concat")
    assert signer.derive_key() == signer.digest_method(signer.salt + b"signer" + b"secret-key").digest()

    # Test with custom salt
    signer = Signer("secret-key", salt="custom-salt", key_derivation="django-concat")
    assert signer.derive_key() == signer.digest_method(b"custom-salt" + b"signer" + b"secret-key").digest()

    # Test with unknown key derivation method
    signer = Signer("secret-key", key_derivation="unknown")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer.derive_key()


# LLM-generated content at query #7
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key()
    expected_key = _lazy_sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with concat key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    derived_key = signer.derive_key()
    expected_key = _lazy_sha1(b"test-salt" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with hmac key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    derived_key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=_lazy_sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert derived_key == expected_key

    # Test with none key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    derived_key = signer.derive_key()
    assert derived_key == b"secret-key"

    # Test with custom secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key("custom-key")
    expected_key = _lazy_sha1(b"test-salt" + b"signer" + b"custom-key").digest()
    assert derived_key == expected_key

    # Test with bytes secret_key
    signer = Signer(b"secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = _lazy_sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with bytes salt
    signer = Signer("secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = _lazy_sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with unknown key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="unknown")
    try:
        signer.derive_key()
        assert False, "Expected TypeError for unknown key derivation"
    except TypeError:
        pass


# LLM-generated content at query #8
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
    base64_sig = base64_encode(b"test-sig")
    assert signer.verify_signature(value, base64_sig) is False

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

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #9
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

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    signature_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, signature_hmac) is True
    assert signer.verify_signature(value, signature_hmac) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    signature_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature_none) is True


# LLM-generated content at query #10
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key_derivation ('django-concat')
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with 'concat' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with 'hmac' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    derived_key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert derived_key == expected_key

    # Test with 'none' key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    derived_key = signer.derive_key()
    assert derived_key == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key("other-secret")
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"other-secret").digest()
    assert derived_key == expected_key

    # Test with bytes secret_key
    signer = Signer(b"secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with bytes salt
    signer = Signer("secret-key", salt=b"test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with invalid key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="invalid")
    with pytest.raises(TypeError):
        signer.derive_key()


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
    base64_sig = base64_encode(b"wrong-sig")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    sig_old = signer_rotated.get_signature(value)  # uses new-key
    # Manually create a signature with old-key
    old_key = signer_rotated.derive_key(want_bytes("old-key"))
    mac = hmac.new(old_key, msg=value, digestmod=signer_rotated.digest_method)
    sig_old_key = base64_encode(mac.digest())
    assert signer_rotated.verify_signature(value, sig_old_key) is True
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    sig = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig) is True


# LLM-generated content at query #12
#--------------------------

```python
def test_Signer_derive_key():
    # Test concat key derivation
    signer = Signer("secret", key_derivation="concat")
    key = signer.derive_key()
    assert key == hashlib.sha1(b"itsdangerous.Signer" + b"secret").digest()

    # Test django-concat key derivation
    signer = Signer("secret", key_derivation="django-concat")
    key = signer.derive_key()
    assert key == hashlib.sha1(b"itsdangerous.Signer" + b"signer" + b"secret").digest()

    # Test hmac key derivation
    signer = Signer("secret", key_derivation="hmac")
    key = signer.derive_key()
    mac = hmac.new(b"secret", digestmod=hashlib.sha1)
    mac.update(b"itsdangerous.Signer")
    assert key == mac.digest()

    # Test none key derivation
    signer = Signer("secret", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret"

    # Test with custom salt
    signer = Signer("secret", salt="custom", key_derivation="concat")
    key = signer.derive_key()
    assert key == hashlib.sha1(b"custom" + b"secret").digest()

    # Test with custom secret_key parameter
    signer = Signer("secret", key_derivation="concat")
    key = signer.derive_key("custom")
    assert key == hashlib.sha1(b"itsdangerous.Signer" + b"custom").digest()

    # Test with bytes secret key
    signer = Signer(b"secret", key_derivation="concat")
    key = signer.derive_key()
    assert key == hashlib.sha1(b"itsdangerous.Signer" + b"secret").digest()

    # Test with bytes salt
    signer = Signer("secret", salt=b"custom", key_derivation="concat")
    key = signer.derive_key()
    assert key == hashlib.sha1(b"custom" + b"secret").digest()

    # Test with invalid key_derivation
    signer = Signer("secret", key_derivation="invalid")
    with pytest.raises(TypeError):
        signer.derive_key()


# LLM-generated content at query #13
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert key == expected_key

    # Test with concat key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert key == expected_key

    # Test with hmac key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert key == expected_key

    # Test with none key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key("another-secret-key")
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"another-secret-key").digest()
    assert key == expected_key

    # Test with bytes secret_key
    signer = Signer(b"secret-key", salt="test-salt")
    key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert key == expected_key

    # Test with bytes salt
    signer = Signer("secret-key", salt=b"test-salt")
    key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert key == expected_key

    # Test with unknown key derivation method
    signer = Signer("secret-key", salt="test-salt", key_derivation="unknown")
    try:
        signer.derive_key()
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key derivation (django-concat)
    signer = Signer("secret")
    assert signer.derive_key() == signer.digest_method(signer.salt + b"signer" + b"secret").digest()

    # Test with concat key derivation
    signer = Signer("secret", key_derivation="concat")
    assert signer.derive_key() == signer.digest_method(signer.salt + b"secret").digest()

    # Test with hmac key derivation
    signer = Signer("secret", key_derivation="hmac")
    mac = hmac.new(b"secret", digestmod=signer.digest_method)
    mac.update(signer.salt)
    assert signer.derive_key() == mac.digest()

    # Test with none key derivation
    signer = Signer("secret", key_derivation="none")
    assert signer.derive_key() == b"secret"

    # Test with specific secret key
    signer = Signer("secret")
    assert signer.derive_key(b"other") == signer.digest_method(signer.salt + b"signer" + b"other").digest()

    # Test with bytes secret key
    signer = Signer(b"secret")
    assert signer.derive_key() == signer.digest_method(signer.salt + b"signer" + b"secret").digest()

    # Test with unknown key derivation
    signer = Signer("secret", key_derivation="unknown")
    with pytest.raises(TypeError):
        signer.derive_key()


# LLM-generated content at query #15
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
    base64_sig = base64_encode(b"wrong-sig")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"rotated-value"
    sig_old = signer_rotated.get_signature(value)  # uses new-key
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Create signature with old key
    old_signer = Signer("old-key", salt=signer_rotated.salt)
    sig_old = old_signer.get_signature(value)
    assert signer_rotated.verify_signature(value, sig_old) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"derivation-test"
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"sha256-test"
    sig = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, sig) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"none-algo-test"
    sig = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig) is True


# LLM-generated content at query #16
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

    # Test with non-existent key in rotation
    assert signer_rotated.verify_signature(value, b"wrong-signature") is False

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


# LLM-generated content at query #17
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key_derivation ("django-concat")
    signer = Signer("secret-key", salt="test-salt")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with "concat" key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    derived_key = signer.derive_key()
    expected_key = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert derived_key == expected_key

    # Test with "hmac" key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    derived_key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert derived_key == expected_key

    # Test with "none" key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    derived_key = signer.derive_key()
    assert derived_key == b"secret-key"

    # Test with a specific secret_key
    signer = Signer(["old-key", "new-key"], salt="test-salt")
    derived_key = signer.derive_key("old-key")
    expected_key = hashlib.sha1(b"test-salt" + b"signer" + b"old-key").digest()
    assert derived_key == expected_key


# LLM-generated content at query #18
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
    base64_sig = base64_encode(b"wrong-sig")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with multiple secret keys (key rotation)
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"rotated-value"
    signature = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, signature) is True

    # Test with old key still valid
    old_signer = Signer("old-key")
    old_sig = old_signer.get_signature(value)
    assert signer_rotated.verify_signature(value, old_sig) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

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


# LLM-generated content at query #19
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

    # Test with multiple keys (key rotation)
    signer_multi = Signer(["old-key", "new-key"])
    value_multi = b"test-value-multi"
    signature_multi = signer_multi.get_signature(value_multi)
    assert signer_multi.verify_signature(value_multi, signature_multi) is True

    # Test with old key
    old_signer = Signer("old-key")
    old_signature = old_signer.get_signature(value_multi)
    assert signer_multi.verify_signature(value_multi, old_signature) is True

    # Test with wrong key
    wrong_signer = Signer("wrong-key")
    wrong_signature = wrong_signer.get_signature(value)
    assert signer.verify_signature(value, wrong_signature) is False

    # Test with different key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    signature_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, signature_hmac) is True
    assert signer.verify_signature(value, signature_hmac) is False

    # Test with different digest method
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    signature_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature_sha256) is True
    assert signer.verify_signature(value, signature_sha256) is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    signature_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature_none) is True
    assert signer.verify_signature(value, signature_none) is False


# LLM-generated content at query #20
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key", salt="test-salt")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature)

    # Test with incorrect signature
    assert not signer.verify_signature(value, b"wrong-signature")

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"wrong-signature"))

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    signature_old = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old)

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_deriv = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        signature_deriv = signer_deriv.get_signature(value)
        assert signer_deriv.verify_signature(value, signature_deriv)

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", salt="test-salt", digest_method=hashlib.sha256)
    signature_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature_sha256)

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    signature_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature_none)


# LLM-generated content at query #21
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
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value, secret_key=b"old-key")
    sig_new = signer_rotated.get_signature(value, secret_key=b"new-key")
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

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


# LLM-generated content at query #22
#--------------------------

```python
def test_Signer_derive_key():
    # Test with default key_derivation ("django-concat")
    signer = Signer("secret-key", salt="test-salt")
    expected_key = signer.digest_method(signer.salt + b"signer" + b"secret-key").digest()
    assert signer.derive_key() == expected_key

    # Test with "concat" key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    expected_key = signer.digest_method(b"test-salt" + b"secret-key").digest()
    assert signer.derive_key() == expected_key

    # Test with "hmac" key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    mac = hmac.new(b"secret-key", digestmod=signer.digest_method)
    mac.update(b"test-salt")
    expected_key = mac.digest()
    assert signer.derive_key() == expected_key

    # Test with "none" key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    assert signer.derive_key() == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    assert signer.derive_key("other-key") == signer.digest_method(b"test-salt" + b"signer" + b"other-key").digest()

    # Test with bytes secret_key
    signer = Signer(b"secret-key", salt=b"test-salt")
    expected_key = signer.digest_method(b"test-salt" + b"signer" + b"secret-key").digest()
    assert signer.derive_key() == expected_key

    # Test with bytes salt
    signer = Signer("secret-key", salt=b"test-salt")
    expected_key = signer.digest_method(b"test-salt" + b"signer" + b"secret-key").digest()
    assert signer.derive_key() == expected_key

    # Test with unknown key_derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="unknown")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer.derive_key()


# LLM-generated content at query #23
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
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

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

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


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
    assert signer.verify_signature(value, b"invalid-sig") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"invalid-sig")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value, secret_key=b"old-key")
    sig_new = signer_rotated.get_signature(value, secret_key=b"new-key")
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True
    assert signer_rotated.verify_signature(value, b"invalid-sig") is False

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


# LLM-generated content at query #25
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
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

    # Test with key rotation (old key)
    old_key = b"old-secret-key"
    new_key = b"new-secret-key"
    signer = Signer([old_key, new_key], salt="test-salt")
    value = b"test-value"
    old_sig = Signer(old_key, salt="test-salt").get_signature(value)
    new_sig = Signer(new_key, salt="test-salt").get_signature(value)
    assert signer.verify_signature(value, old_sig) is True
    assert signer.verify_signature(value, new_sig) is True

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

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


# LLM-generated content at query #26
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
    assert derived_key != derived_key_concat

    # Test with hmac key derivation
    signer_hmac = Signer("secret-key", key_derivation="hmac")
    derived_key_hmac = signer_hmac.derive_key()
    assert isinstance(derived_key_hmac, bytes)
    assert len(derived_key_hmac) > 0
    assert derived_key != derived_key_hmac

    # Test with none key derivation
    signer_none = Signer("secret-key", key_derivation="none")
    derived_key_none = signer_none.derive_key()
    assert derived_key_none == b"secret-key"

    # Test with specific secret_key parameter
    custom_key = signer.derive_key("custom-secret")
    assert isinstance(custom_key, bytes)
    assert len(custom_key) > 0
    assert custom_key != derived_key

    # Test with bytes secret key
    signer_bytes = Signer(b"secret-key")
    derived_key_bytes = signer_bytes.derive_key()
    assert isinstance(derived_key_bytes, bytes)
    assert len(derived_key_bytes) > 0

    # Test with list of secret keys (key rotation)
    signer_rotation = Signer(["old-key", "new-key"])
    derived_key_rotation = signer_rotation.derive_key()
    assert isinstance(derived_key_rotation, bytes)
    assert len(derived_key_rotation) > 0

    # Test that unknown key derivation raises TypeError
    try:
        signer_bad = Signer("secret-key", key_derivation="unknown")
        signer_bad.derive_key()
        assert False, "Expected TypeError for unknown key derivation"
    except TypeError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    wrong_signature = b"wrong-signature"
    assert signer.verify_signature(value, wrong_signature) is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"some-signature")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    invalid_base64_sig = b"invalid-base64!"
    assert signer.verify_signature(value, invalid_base64_sig) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"rotated-value"
    signature_old = signer_rotated.get_signature(value)
    # Change the key to simulate rotation
    signer_rotated.secret_keys = ["new-key"]
    assert signer_rotated.verify_signature(value, signature_old) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"derivation-test"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with different digest methods
    def custom_digest(string=b""):
        return hashlib.sha256(string)

    signer_custom = Signer("secret-key", digest_method=custom_digest)
    value = b"custom-digest-test"
    signature = signer_custom.get_signature(value)
    assert signer_custom.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"none-algorithm-test"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True


# LLM-generated content at query #28
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

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    old_sig = signer_rotated.get_signature(value, secret_key="old-key")
    new_sig = signer_rotated.get_signature(value, secret_key="new-key")
    assert signer_rotated.verify_signature(value, old_sig) is True
    assert signer_rotated.verify_signature(value, new_sig) is True
    assert signer_rotated.verify_signature(value, b"wrong-sig") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True
        assert signer.verify_signature(value, b"wrong-sig") is False

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    signature_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature_sha256) is True
    assert signer_sha256.verify_signature(value, b"wrong-sig") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    assert signer_none.verify_signature(value, b"") is True
    assert signer_none.verify_signature(value, b"any-sig") is False


# LLM-generated content at query #29
#--------------------------

```python
def test_Signer_derive_key():
    # Test default key derivation (django-concat)
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key()
    expected = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert key == expected

    # Test concat key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="concat")
    key = signer.derive_key()
    expected = hashlib.sha1(b"test-salt" + b"secret-key").digest()
    assert key == expected

    # Test hmac key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod=hashlib.sha1)
    mac.update(b"test-salt")
    expected = mac.digest()
    assert key == expected

    # Test none key derivation
    signer = Signer("secret-key", salt="test-salt", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret-key"

    # Test with specific secret_key parameter
    signer = Signer("secret-key", salt="test-salt")
    key = signer.derive_key("specific-key")
    expected = hashlib.sha1(b"test-salt" + b"signer" + b"specific-key").digest()
    assert key == expected

    # Test with bytes secret key
    signer = Signer(b"secret-key", salt=b"test-salt")
    key = signer.derive_key()
    expected = hashlib.sha1(b"test-salt" + b"signer" + b"secret-key").digest()
    assert key == expected

    # Test with custom digest method
    def custom_digest(x):
        return hashlib.sha256(x)
    signer = Signer("secret-key", salt="test-salt", digest_method=custom_digest)
    key = signer.derive_key()
    expected = hashlib.sha256(b"test-salt" + b"signer" + b"secret-key").digest()
    assert key == expected

    # Test with key rotation (multiple secret keys)
    signer = Signer(["old-key", "new-key"], salt="test-salt")
    key = signer.derive_key()
    expected = hashlib.sha1(b"test-salt" + b"signer" + b"new-key").digest()
    assert key == expected

    # Test with specific old key in rotation
    signer = Signer(["old-key", "new-key"], salt="test-salt")
    key = signer.derive_key("old-key")
    expected = hashlib.sha1(b"test-salt" + b"signer" + b"old-key").digest()
    assert key == expected

    # Test with None salt (should use default)
    signer = Signer("secret-key", salt=None)
    key = signer.derive_key()
    expected = hashlib.sha1(b"itsdangerous.Signer" + b"signer" + b"secret-key").digest()
    assert key == expected


# LLM-generated content at query #30
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
    assert signer.verify_signature(value, base64_encode(b"wrong")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    old_sig = signer_rotated.get_signature(value, secret_key=b"old-key")
    new_sig = signer_rotated.get_signature(value, secret_key=b"new-key")
    assert signer_rotated.verify_signature(value, old_sig) is True
    assert signer_rotated.verify_signature(value, new_sig) is True
    assert signer_rotated.verify_signature(value, b"wrong") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True
        assert signer.verify_signature(value, b"wrong") is False

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    signature_sha256 = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature_sha256) is True
    assert signer_sha256.verify_signature(value, b"wrong") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    signature_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature_none) is True
    assert signer_none.verify_signature(value, b"wrong") is False


# LLM-generated content at query #31
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

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    old_signer = Signer("old-key")
    new_signer = Signer("new-key")
    old_value = b"old-value"
    new_value = b"new-value"
    old_signature = old_signer.get_signature(old_value)
    new_signature = new_signer.get_signature(new_value)

    # Old key should verify old signature
    assert signer_rotated.verify_signature(old_value, old_signature) is True
    # New key should verify new signature
    assert signer_rotated.verify_signature(new_value, new_signature) is True
    # Old key should not verify new signature
    assert signer_rotated.verify_signature(new_value, old_signature) is False
    # New key should not verify old signature
    assert signer_rotated.verify_signature(old_value, new_signature) is False

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
    assert signer_none.verify_signature(value, b"") is True  # empty signature is valid for NoneAlgorithm


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
    assert signer.verify_signature(value, base64_encode(b"wrong-signature")) is False

    # Test with different value
    assert signer.verify_signature(b"different-value", signature) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    signer_rotated.secret_keys = ["new-key"]  # Simulate key rotation
    assert signer_rotated.verify_signature(value, signature_old) is False

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

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False


# LLM-generated content at query #33
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


# LLM-generated content at query #34
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
    base64_sig = base64_encode(b"wrong-sig")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"rotated-value"
    signature_old = signer_rotated.get_signature(value)
    # Change to new key and verify old signature still works
    signer_rotated.secret_keys = ["old-key", "new-key"]
    assert signer_rotated.verify_signature(value, signature_old) is True
    # Verify new signature works
    signature_new = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, signature_new) is True

    # Test with different key derivation methods
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_deriv = Signer("secret-key", key_derivation=derivation)
        value = b"derivation-test"
        signature = signer_deriv.get_signature(value)
        assert signer_deriv.verify_signature(value, signature) is True

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


# LLM-generated content at query #35
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with valid signature
    signer = Signer("secret-key")
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with invalid signature
    assert signer.verify_signature(value, b"invalid_sig") is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"invalid_sig")) is False

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

    # Test with empty value
    assert signer.verify_signature(b"", signer.get_signature(b"")) is True

    # Test with empty signature
    assert signer.verify_signature(value, b"") is False


# LLM-generated content at query #36
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

    # Test with wrong value
    assert signer.verify_signature(b"wrong-value", signature) is False

    # Test with base64 encoded signature
    assert signer.verify_signature(value, base64_encode(b"invalid")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    old_signer = Signer("old-key")
    new_signer = Signer("new-key")
    old_value = b"old-value"
    new_value = b"new-value"
    old_signature = old_signer.get_signature(old_value)
    new_signature = new_signer.get_signature(new_value)

    assert signer_rotated.verify_signature(old_value, old_signature) is True
    assert signer_rotated.verify_signature(new_value, new_signature) is True
    assert signer_rotated.verify_signature(old_value, new_signature) is False
    assert signer_rotated.verify_signature(new_value, old_signature) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True


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
    base64_sig = base64_encode(b"wrong")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    assert signer.verify_signature(value, b"invalid-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"rotated-value"
    signature_old = signer_rotated.get_signature(value)
    # Change to new key and verify old signature still works
    signer_rotated.secret_keys = ["old-key", "new-key"]
    assert signer_rotated.verify_signature(value, signature_old) is True
    # Verify new signature works
    signature_new = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, signature_new) is True

    # Test with different key derivation methods
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=derivation)
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
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

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


# LLM-generated content at query #39
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

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    sig_old = signer_rotated.get_signature(value, secret_key=b"old-key")
    sig_new = signer_rotated.get_signature(value, secret_key=b"new-key")
    assert signer_rotated.verify_signature(value, sig_old) is True
    assert signer_rotated.verify_signature(value, sig_new) is True
    assert signer_rotated.verify_signature(value, b"invalid-sig") is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_kd = Signer("secret-key", key_derivation=key_derivation)
        sig_kd = signer_kd.get_signature(value)
        assert signer_kd.verify_signature(value, sig_kd) is True
        assert signer_kd.verify_signature(value, b"invalid-sig") is False

    # Test with different digest methods
    for digest_method in [hashlib.sha256, hashlib.sha512]:
        signer_dm = Signer("secret-key", digest_method=digest_method)
        sig_dm = signer_dm.get_signature(value)
        assert signer_dm.verify_signature(value, sig_dm) is True
        assert signer_dm.verify_signature(value, b"invalid-sig") is False

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True
    assert signer_none.verify_signature(value, b"invalid-sig") is False


# LLM-generated content at query #40
#--------------------------

```python
def test_Signer_verify_signature():
    # Setup
    secret_key = b"secret"
    salt = b"test_salt"
    signer = Signer(secret_key, salt=salt)
    value = b"test_value"
    signature = signer.get_signature(value)

    # Test valid signature
    assert signer.verify_signature(value, signature) is True

    # Test invalid signature
    assert signer.verify_signature(value, b"invalid_signature") is False

    # Test with different value
    assert signer.verify_signature(b"different_value", signature) is False

    # Test with base64 encoded string signature
    assert signer.verify_signature(value, base64_encode(b"invalid")) is False

    # Test with key rotation
    secret_keys = [b"old_secret", b"new_secret"]
    signer_rotated = Signer(secret_keys, salt=salt)
    new_signature = signer_rotated.get_signature(value)

    # Should verify with newest key
    assert signer_rotated.verify_signature(value, new_signature) is True

    # Should not verify with old key's signature
    old_signer = Signer(b"old_secret", salt=salt)
    old_signature = old_signer.get_signature(value)
    assert signer_rotated.verify_signature(value, old_signature) is True  # old key still works

    # Test with different key derivation
    signer_hmac = Signer(secret_key, salt=salt, key_derivation="hmac")
    hmac_signature = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, hmac_signature) is True
    assert signer.verify_signature(value, hmac_signature) is False  # different derivation

    # Test with NoneAlgorithm
    none_signer = Signer(secret_key, salt=salt, algorithm=NoneAlgorithm())
    none_signature = none_signer.get_signature(value)
    assert none_signer.verify_signature(value, none_signature) is True


# LLM-generated content at query #41
#--------------------------

```python
def test_Signer_verify_signature():
    # Test with correct signature
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

    # Test with incorrect signature
    wrong_signature = b"wrong-signature"
    assert signer.verify_signature(value, wrong_signature) is False

    # Test with base64 encoded signature
    base64_sig = base64_encode(b"some-signature")
    assert signer.verify_signature(value, base64_sig) is False

    # Test with invalid base64 signature
    invalid_base64 = b"invalid-base64!"
    assert signer.verify_signature(value, invalid_base64) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"])
    value = b"rotated-value"
    signature_old = signer_rotated.get_signature(value)
    # Simulate signing with old key by creating a new signer with only old key
    old_signer = Signer("old-key", salt=signer_rotated.salt)
    signature_old_actual = old_signer.get_signature(value)
    assert signer_rotated.verify_signature(value, signature_old_actual) is True
    # New key should also work
    signature_new = signer_rotated.get_signature(value)
    assert signer_rotated.verify_signature(value, signature_new) is True

    # Test with different key derivations
    for derivation in ["concat", "django-concat", "hmac", "none"]:
        signer_deriv = Signer("secret-key", key_derivation=derivation)
        value = b"derivation-test"
        signature = signer_deriv.get_signature(value)
        assert signer_deriv.verify_signature(value, signature) is True

    # Test with different digest methods
    signer_sha256 = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"sha256-test"
    signature = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    none_algo = NoneAlgorithm()
    signer_none = Signer("secret-key", algorithm=none_algo)
    value = b"none-algo-test"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True


# LLM-generated content at query #42
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
    assert signer.verify_signature(value, b"not-base64!") is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    # Change to new key
    signer_rotated.secret_keys = ["old-key", "new-key"]
    assert signer_rotated.verify_signature(value, signature_old) is True

    # Test with different key derivation
    signer_hmac = Signer("secret-key", salt="test-salt", key_derivation="hmac")
    value = b"test-value"
    signature = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True


# LLM-generated content at query #43
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

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)  # uses new-key
    # Manually create signature with old key
    old_key = signer_rotated.derive_key(want_bytes("old-key"))
    hmac_old = hmac.new(old_key, msg=value, digestmod=signer_rotated.digest_method)
    sig_old = base64_encode(hmac_old.digest())
    assert signer_rotated.verify_signature(value, sig_old) is True
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


# LLM-generated content at query #44
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
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

    # Test with key rotation
    signer_rotated = Signer(["old-key", "new-key"], salt="test-salt")
    value = b"test-value"
    signature_old = signer_rotated.get_signature(value)
    # Change the key to simulate rotation
    signer_rotated.secret_keys = ["new-key"]
    assert signer_rotated.verify_signature(value, signature_old) is False

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", salt="test-salt", key_derivation=key_derivation)
        value = b"test-value"
        signature = signer.get_signature(value)
        assert signer.verify_signature(value, signature) is True

    # Test with NoneAlgorithm
    signer_none = Signer("secret-key", salt="test-salt", algorithm=NoneAlgorithm())
    value = b"test-value"
    signature = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, signature) is True

    # Test with HMACAlgorithm and different digest methods
    signer_sha256 = Signer(
        "secret-key",
        salt="test-salt",
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    value = b"test-value"
    signature = signer_sha256.get_signature(value)
    assert signer_sha256.verify_signature(value, signature) is True


# LLM-generated content at query #45
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
    assert signer.verify_signature(value, base64_encode(b"wrong-sig")) is False

    # Test with key rotation
    signer = Signer(["old-key", "new-key"])
    sig_old = signer.get_signature(value, secret_key="old-key")
    sig_new = signer.get_signature(value, secret_key="new-key")
    assert signer.verify_signature(value, sig_old) is True
    assert signer.verify_signature(value, sig_new) is True

    # Test with different key derivation methods
    for key_derivation in ["concat", "django-concat", "hmac", "none"]:
        signer = Signer("secret-key", key_derivation=key_derivation)
        sig = signer.get_signature(value)
        assert signer.verify_signature(value, sig) is True

    # Test with different digest methods
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with custom algorithm
    signer = Signer("secret-key", algorithm=NoneAlgorithm())
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

    # Test with empty signature
    assert signer.verify_signature(value, b"") is False


