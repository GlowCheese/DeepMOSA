####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_derive_key_with_concat_derivation():
    signer = Signer(b"secret", key_derivation="concat")
    assert signer.derive_key() == hashlib.sha1(b"itsdangerous.Signer" + b"secret").digest()

def test_derive_key_with_django_concat_derivation():
    signer = Signer(b"secret", key_derivation="django-concat")
    assert signer.derive_key() == hashlib.sha1(b"itsdangerous.Signer" + b"signer" + b"secret").digest()

def test_derive_key_with_hmac_derivation():
    signer = Signer(b"secret", key_derivation="hmac")
    mac = hmac.new(b"secret", digestmod=hashlib.sha1)
    mac.update(b"itsdangerous.Signer")
    assert signer.derive_key() == mac.digest()

def test_derive_key_with_none_derivation():
    signer = Signer(b"secret", key_derivation="none")
    assert signer.derive_key() == b"secret"

def test_derive_key_with_custom_salt():
    signer = Signer(b"secret", salt=b"custom", key_derivation="concat")
    assert signer.derive_key() == hashlib.sha1(b"custom" + b"secret").digest()

def test_derive_key_with_custom_secret_key():
    signer = Signer(b"secret", key_derivation="concat")
    assert signer.derive_key(b"custom") == hashlib.sha1(b"itsdangerous.Signer" + b"custom").digest()

def test_derive_key_with_invalid_derivation():
    signer = Signer(b"secret", key_derivation="invalid")
    try:
        signer.derive_key()
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_list_secret_keys():
    signer = Signer(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_salt():
    signer = Signer("my-secret-key", salt="custom-salt")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_sep():
    signer = Signer("my-secret-key", sep="|")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"|"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("my-secret-key", key_derivation="hmac")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "hmac"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("my-secret-key", digest_method=hashlib.sha256)
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_algorithm():
    class CustomAlgorithm(SigningAlgorithm):
        def get_signature(self, key: bytes, value: bytes) -> bytes:
            return b"custom-signature"

    algorithm = CustomAlgorithm()
    signer = Signer("my-secret-key", algorithm=algorithm)
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm == algorithm

def test_signer_constructor_with_none_salt():
    signer = Signer("my-secret-key", salt=None)
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_invalid_sep():
    import pytest
    with pytest.raises(ValueError):
        Signer("my-secret-key", sep="a")

def test_signer_constructor_with_invalid_key_derivation():
    import pytest
    with pytest.raises(TypeError):
        Signer("my-secret-key", key_derivation="invalid")


# LLM-generated content at query #3
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_iterable_secret_key():
    signer = Signer(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"custom-salt"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep="|")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret-key", key_derivation="hmac")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert signer.algorithm is algorithm

def test_signer_constructor_with_invalid_separator():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret-key", sep="-")


# LLM-generated content at query #4
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = Signer("secret-key")
    signed_value = signer.sign("value")
    assert signer.unsign(signed_value) == b"value"

def test_unsign_with_invalid_signature():
    signer = Signer("secret-key")
    signed_value = b"value.sep.invalid_signature"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_unsign_with_missing_separator():
    signer = Signer("secret-key")
    signed_value = b"value_without_separator"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadSignature exception"
    except BadSignature as e:
        assert "No b'.' found in value" in str(e)

def test_unsign_with_wrong_secret_key():
    signer1 = Signer("secret-key-1")
    signer2 = Signer("secret-key-2")
    signed_value = signer1.sign("value")
    try:
        signer2.unsign(signed_value)
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_unsign_with_bytes_input():
    signer = Signer("secret-key")
    signed_value = signer.sign(b"value")
    assert signer.unsign(signed_value) == b"value"

def test_unsign_with_string_input():
    signer = Signer("secret-key")
    signed_value = signer.sign("value")
    assert signer.unsign(signed_value.decode()) == b"value"


# LLM-generated content at query #5
#--------------------------

```python
def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    invalid_signature = b"invalid-signature"
    assert signer.verify_signature(value, invalid_signature) is False

def test_verify_signature_with_malformed_base64():
    signer = Signer("secret-key")
    value = b"test-value"
    malformed_signature = b"malformed!!!"
    assert signer.verify_signature(value, malformed_signature) is False

def test_verify_signature_with_rotated_keys():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    old_signer = Signer("old-key")
    old_signature = old_signer.get_signature(value)
    assert signer.verify_signature(value, old_signature) is True

def test_verify_signature_with_newest_key():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    new_signature = signer.get_signature(value)
    assert signer.verify_signature(value, new_signature) is True


# LLM-generated content at query #6
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_iterable_secret_keys():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"custom-salt"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep="|")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret-key", key_derivation="hmac")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_digest_method():
    signer = Signer("secret-key", digest_method=_lazy_sha256)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(_lazy_sha256)
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert signer.algorithm == algorithm

def test_signer_constructor_with_invalid_sep():
    try:
        Signer("secret-key", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "The given separator cannot be used because it may be contained in the signature itself. ASCII letters, digits, and '-_=' must not be used."

def test_signer_constructor_with_none_salt():
    signer = Signer("secret-key", salt=None)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #7
#--------------------------

```python
def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    invalid_signature = b"invalid-signature"
    assert signer.verify_signature(value, invalid_signature) is False

def test_verify_signature_with_wrong_value():
    signer = Signer("secret-key")
    value = b"test-value"
    other_value = b"other-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(other_value, signature) is False

def test_verify_signature_with_malformed_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    malformed_signature = b"malformed!!!"
    assert signer.verify_signature(value, malformed_signature) is False

def test_verify_signature_with_key_rotation():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = Signer("old-key").get_signature(value)
    signature_new = Signer("new-key").get_signature(value)
    assert signer.verify_signature(value, signature_old) is True
    assert signer.verify_signature(value, signature_new) is True

def test_verify_signature_with_different_key_derivation():
    signer_concat = Signer("secret-key", key_derivation="concat")
    signer_django = Signer("secret-key", key_derivation="django-concat")
    value = b"test-value"
    signature_concat = signer_concat.get_signature(value)
    signature_django = signer_django.get_signature(value)
    assert signer_concat.verify_signature(value, signature_concat) is True
    assert signer_django.verify_signature(value, signature_django) is True
    assert signer_concat.verify_signature(value, signature_django) is False


# LLM-generated content at query #8
#--------------------------

```python
def test_signer_constructor_with_string_key():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_key():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep="|")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"|"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_digest_method():
    signer = Signer("secret-key", digest_method=_lazy_sha256)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(_lazy_sha256)
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert signer.algorithm == algorithm

def test_signer_constructor_with_key_list():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_invalid_sep():
    try:
        Signer("secret-key", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == (
            "The given separator cannot be used because it may be"
            " contained in the signature itself. ASCII letters,"
            " digits, and '-_=' must not be used."
        )

def test_signer_constructor_with_none_salt():
    signer = Signer("secret-key", salt=None)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #9
#--------------------------

```python
def test_verify_signature_returns_false_on_base64_decode_exception():
    signer = Signer("secret-key")
    assert not signer.verify_signature("value", "invalid-base64-sig")


# LLM-generated content at query #10
#--------------------------

```python
def test_verify_signature_with_invalid_base64_sig():
    signer = Signer("secret-key")
    assert signer.verify_signature("value", "invalid-base64") is False


# LLM-generated content at query #11
#--------------------------

```python
def test_verify_signature_with_correct_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

def test_verify_signature_with_incorrect_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    wrong_signature = b"wrong-signature"
    assert signer.verify_signature(value, wrong_signature) is False

def test_verify_signature_with_invalid_base64():
    signer = Signer("secret-key")
    value = b"test-value"
    invalid_base64 = b"invalid-base64!"
    assert signer.verify_signature(value, invalid_base64) is False

def test_verify_signature_with_different_value():
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    different_value = b"different-value"
    assert signer.verify_signature(different_value, signature) is False

def test_verify_signature_with_key_rotation():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature_old = Signer("old-key").get_signature(value)
    signature_new = Signer("new-key").get_signature(value)
    assert signer.verify_signature(value, signature_old) is True
    assert signer.verify_signature(value, signature_new) is True

def test_verify_signature_with_wrong_key():
    signer = Signer("secret-key")
    other_signer = Signer("other-key")
    value = b"test-value"
    signature = other_signer.get_signature(value)
    assert signer.verify_signature(value, signature) is False


# LLM-generated content at query #12
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_iterable_secret_key():
    signer = Signer(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep="|")
    assert signer.sep == b"|"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_custom_digest_method():
    custom_digest = hashlib.sha256
    signer = Signer("secret-key", digest_method=custom_digest)
    assert signer.digest_method == custom_digest
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    custom_algorithm = HMACAlgorithm(hashlib.sha256)
    signer = Signer("secret-key", algorithm=custom_algorithm)
    assert signer.algorithm == custom_algorithm

def test_signer_constructor_with_none_salt():
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_invalid_sep():
    with pytest.raises(ValueError):
        Signer("secret-key", sep="a")


# LLM-generated content at query #13
#--------------------------

```python
def test_verify_signature_returns_false_on_invalid_base64():
    signer = Signer("secret-key")
    assert signer.verify_signature("value", "invalid-base64") is False


# LLM-generated content at query #14
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_list_secret_key():
    signer = Signer(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"custom-salt"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep="|")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_digest_method():
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert signer.algorithm == algorithm

def test_signer_constructor_with_invalid_sep():
    with pytest.raises(ValueError):
        Signer("secret-key", sep="a")


# LLM-generated content at query #15
#--------------------------

```python
def test_verify_signature_returns_false_on_invalid_base64():
    signer = Signer("secret-key")
    assert signer.verify_signature("value", "invalid-base64") is False


# LLM-generated content at query #16
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_list_secret_keys():
    signer = Signer(["key1", "key2", "key3"])
    assert signer.secret_keys == [b"key1", b"key2", b"key3"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_salt():
    signer = Signer("my-secret-key", salt="custom-salt")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_sep():
    signer = Signer("my-secret-key", sep="|")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"|"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("my-secret-key", key_derivation="concat")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("my-secret-key", digest_method=hashlib.sha256)
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha256
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_algorithm():
    class CustomAlgorithm(SigningAlgorithm):
        def get_signature(self, key: bytes, value: bytes) -> bytes:
            return b"custom-signature"

    algorithm = CustomAlgorithm()
    signer = Signer("my-secret-key", algorithm=algorithm)
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm == algorithm

def test_signer_constructor_with_invalid_sep():
    import pytest
    with pytest.raises(ValueError):
        Signer("my-secret-key", sep="a")


# LLM-generated content at query #17
#--------------------------

```python
def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    invalid_signature = b"invalid-signature"
    assert signer.verify_signature(value, invalid_signature) is False

def test_verify_signature_with_wrong_value():
    signer = Signer("secret-key")
    value = b"test-value"
    other_value = b"other-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(other_value, signature) is False

def test_verify_signature_with_base64_decode_error():
    signer = Signer("secret-key")
    value = b"test-value"
    invalid_base64 = b"invalid-base64!"
    assert signer.verify_signature(value, invalid_base64) is False

def test_verify_signature_with_rotated_keys():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    old_signer = Signer("old-key")
    old_signature = old_signer.get_signature(value)
    assert signer.verify_signature(value, old_signature) is True

def test_verify_signature_with_newest_key():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    new_signature = signer.get_signature(value)
    assert signer.verify_signature(value, new_signature) is True


# LLM-generated content at query #18
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_iterable_secret_key():
    signer = Signer(["secret1", "secret2"])
    assert signer.secret_keys == [b"secret1", b"secret2"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt="custom-salt")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"custom-salt"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep="|")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_digest_method():
    digest_method = staticmethod(lambda x: x)
    signer = Signer("secret", digest_method=digest_method)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert signer.algorithm == algorithm

def test_signer_constructor_with_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        Signer("secret", sep="a")

def test_signer_constructor_with_none_salt():
    signer = Signer("secret", salt=None)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"hello"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"hello"
    sig = b"invalid-signature"
    assert signer.verify_signature(value, sig) is False

def test_verify_signature_with_wrong_value():
    signer = Signer("secret-key")
    value = b"hello"
    sig = signer.get_signature(b"world")
    assert signer.verify_signature(value, sig) is False

def test_verify_signature_with_malformed_signature():
    signer = Signer("secret-key")
    value = b"hello"
    sig = b"malformed$$$"
    assert signer.verify_signature(value, sig) is False

def test_verify_signature_with_key_rotation():
    signer = Signer(["old-key", "new-key"])
    value = b"hello"
    sig_old = Signer("old-key").get_signature(value)
    sig_new = Signer("new-key").get_signature(value)
    assert signer.verify_signature(value, sig_old) is True
    assert signer.verify_signature(value, sig_new) is True

def test_verify_signature_with_expired_key():
    signer = Signer(["expired-key", "current-key"])
    value = b"hello"
    sig_expired = Signer("expired-key").get_signature(value)
    sig_current = Signer("current-key").get_signature(value)
    assert signer.verify_signature(value, sig_expired) is True
    assert signer.verify_signature(value, sig_current) is True


# LLM-generated content at query #2
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_list_secret_key():
    signer = Signer(["secret1", "secret2"])
    assert signer.secret_keys == [b"secret1", b"secret2"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt="custom_salt")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"custom_salt"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep="|")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_digest_method():
    from hashlib import sha256
    signer = Signer("secret", digest_method=sha256)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is algorithm

def test_signer_constructor_with_invalid_separator():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret", sep="a")

def test_signer_constructor_with_none_salt():
    signer = Signer("secret", salt=None)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None


# LLM-generated content at query #3
#--------------------------

```python
def test_derive_key_with_concat_derivation():
    signer = Signer(b"secret", key_derivation="concat")
    assert signer.derive_key() == hashlib.sha1(b"itsdangerous.Signer" + b"secret").digest()

def test_derive_key_with_django_concat_derivation():
    signer = Signer(b"secret", key_derivation="django-concat")
    assert signer.derive_key() == hashlib.sha1(b"itsdangerous.Signer" + b"signer" + b"secret").digest()

def test_derive_key_with_hmac_derivation():
    signer = Signer(b"secret", key_derivation="hmac")
    mac = hmac.new(b"secret", digestmod=hashlib.sha1)
    mac.update(b"itsdangerous.Signer")
    assert signer.derive_key() == mac.digest()

def test_derive_key_with_none_derivation():
    signer = Signer(b"secret", key_derivation="none")
    assert signer.derive_key() == b"secret"

def test_derive_key_with_custom_salt():
    signer = Signer(b"secret", salt="custom", key_derivation="concat")
    assert signer.derive_key() == hashlib.sha1(b"custom" + b"secret").digest()

def test_derive_key_with_custom_secret_key():
    signer = Signer(b"secret", key_derivation="concat")
    assert signer.derive_key(b"custom") == hashlib.sha1(b"itsdangerous.Signer" + b"custom").digest()

def test_derive_key_with_invalid_derivation():
    signer = Signer(b"secret", key_derivation="invalid")
    try:
        signer.derive_key()
        assert False, "Expected TypeError for invalid key derivation"
    except TypeError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    invalid_signature = b"invalid-signature"
    assert signer.verify_signature(value, invalid_signature) is False

def test_verify_signature_with_wrong_value():
    signer = Signer("secret-key")
    value = b"test-value"
    other_value = b"other-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(other_value, signature) is False

def test_verify_signature_with_malformed_base64():
    signer = Signer("secret-key")
    value = b"test-value"
    malformed_signature = b"malformed-base64!!!"
    assert signer.verify_signature(value, malformed_signature) is False

def test_verify_signature_with_old_secret_key():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    old_signer = Signer("old-key")
    signature = old_signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

def test_verify_signature_with_new_secret_key():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True


# LLM-generated content at query #5
#--------------------------

```python
def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"hello"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"hello"
    invalid_sig = b"invalid"
    assert signer.verify_signature(value, invalid_sig) is False

def test_verify_signature_with_wrong_value():
    signer = Signer("secret-key")
    value = b"hello"
    sig = signer.get_signature(value)
    wrong_value = b"world"
    assert signer.verify_signature(wrong_value, sig) is False

def test_verify_signature_with_malformed_base64():
    signer = Signer("secret-key")
    value = b"hello"
    malformed_sig = b"not-base64!"
    assert signer.verify_signature(value, malformed_sig) is False

def test_verify_signature_with_rotated_keys():
    signer = Signer(["old-key", "new-key"])
    value = b"hello"
    sig_old = Signer("old-key").get_signature(value)
    sig_new = Signer("new-key").get_signature(value)
    assert signer.verify_signature(value, sig_old) is True
    assert signer.verify_signature(value, sig_new) is True

def test_verify_signature_with_string_input():
    signer = Signer("secret-key")
    value = "hello"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True


# LLM-generated content at query #6
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_list_secret_key():
    signer = Signer(["secret1", "secret2"])
    assert signer.secret_keys == [b"secret1", b"secret2"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt="custom_salt")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"custom_salt"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep="|")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_digest_method():
    from hashlib import sha256
    signer = Signer("secret", digest_method=sha256)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert signer.algorithm is algorithm

def test_signer_constructor_with_invalid_sep():
    try:
        Signer("secret", sep="a")
    except ValueError as e:
        assert str(e) == "The given separator cannot be used because it may be contained in the signature itself. ASCII letters, digits, and '-_=' must not be used."
    else:
        assert False, "Expected ValueError"


# LLM-generated content at query #7
#--------------------------

```python
def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    invalid_signature = b"invalid-signature"
    assert signer.verify_signature(value, invalid_signature) is False

def test_verify_signature_with_wrong_value():
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    wrong_value = b"wrong-value"
    assert signer.verify_signature(wrong_value, signature) is False

def test_verify_signature_with_malformed_base64():
    signer = Signer("secret-key")
    value = b"test-value"
    malformed_signature = b"malformed-base64!!!"
    assert signer.verify_signature(value, malformed_signature) is False

def test_verify_signature_with_rotated_keys():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    old_signer = Signer("old-key")
    old_signature = old_signer.get_signature(value)
    assert signer.verify_signature(value, old_signature) is True

def test_verify_signature_with_newest_key():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    new_signature = signer.get_signature(value)
    assert signer.verify_signature(value, new_signature) is True


# LLM-generated content at query #8
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_iterable_secret_keys():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"custom-salt"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep="|")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    class CustomAlgorithm(SigningAlgorithm):
        def get_signature(self, key: bytes, value: bytes) -> bytes:
            return b"custom-signature"

    signer = Signer("secret-key", algorithm=CustomAlgorithm())
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, CustomAlgorithm)

def test_signer_constructor_with_invalid_separator():
    try:
        Signer("secret-key", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "The given separator cannot be used because it may be contained in the signature itself. ASCII letters, digits, and '-_=' must not be used."


# LLM-generated content at query #9
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_iterable_secret_key():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"custom-salt"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep="|")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert signer.algorithm is algorithm

def test_signer_constructor_with_invalid_separator():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret-key", sep="a")


# LLM-generated content at query #10
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep="|")
    assert signer.sep == b"|"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method():
    custom_digest = staticmethod(lambda x: x)
    signer = Signer("secret-key", digest_method=custom_digest)
    assert signer.digest_method == custom_digest

def test_signer_constructor_with_custom_algorithm():
    custom_algorithm = HMACAlgorithm()
    signer = Signer("secret-key", algorithm=custom_algorithm)
    assert signer.algorithm == custom_algorithm

def test_signer_constructor_with_iterable_secret_keys():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_invalid_separator():
    try:
        Signer("secret-key", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot be used" in str(e)

def test_signer_constructor_with_none_salt():
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #11
#--------------------------

```python
def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"value"
    signature = b"invalid-signature"
    assert signer.verify_signature(value, signature) is False

def test_verify_signature_with_wrong_value():
    signer = Signer("secret-key")
    value = b"value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(b"wrong-value", signature) is False

def test_verify_signature_with_malformed_signature():
    signer = Signer("secret-key")
    value = b"value"
    signature = b"malformed!!!"
    assert signer.verify_signature(value, signature) is False

def test_verify_signature_with_rotated_keys():
    signer = Signer(["old-key", "new-key"])
    value = b"value"
    signature_old = Signer("old-key").get_signature(value)
    signature_new = Signer("new-key").get_signature(value)
    assert signer.verify_signature(value, signature_old) is True
    assert signer.verify_signature(value, signature_new) is True

def test_verify_signature_with_string_value():
    signer = Signer("secret-key")
    value = "value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

def test_verify_signature_with_string_signature():
    signer = Signer("secret-key")
    value = b"value"
    signature = signer.get_signature(value).decode("ascii")
    assert signer.verify_signature(value, signature) is True


# LLM-generated content at query #12
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_list_secret_key():
    signer = Signer(["secret1", "secret2"])
    assert signer.secret_keys == [b"secret1", b"secret2"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt="custom_salt")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"custom_salt"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep="|")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is algorithm

def test_signer_constructor_with_invalid_sep():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret", sep="a")

def test_signer_constructor_with_none_salt():
    signer = Signer("secret", salt=None)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None


# LLM-generated content at query #13
#--------------------------

```python
def test_verify_signature_with_invalid_base64():
    signer = Signer("secret-key")
    assert not signer.verify_signature("value", "invalid-base64")


# LLM-generated content at query #14
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_iterable_secret_key():
    signer = Signer(["secret1", "secret2"])
    assert signer.secret_keys == [b"secret1", b"secret2"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt="custom_salt")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"custom_salt"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep="|")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_digest_method():
    from hashlib import sha256
    signer = Signer("secret", digest_method=sha256)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert signer.algorithm is algorithm

def test_signer_constructor_with_invalid_sep():
    try:
        Signer("secret", sep="a")
    except ValueError as e:
        assert str(e) == "The given separator cannot be used because it may be contained in the signature itself. ASCII letters, digits, and '-_=' must not be used."

def test_signer_constructor_with_none_salt():
    signer = Signer("secret", salt=None)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #15
#--------------------------

```python
def test_verify_signature_returns_false_on_base64_decode_exception():
    signer = Signer("secret")
    result = signer.verify_signature("value", "invalid_base64")
    assert result is False


# LLM-generated content at query #16
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer(secret_key="my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(secret_key=b"my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_list_secret_key():
    signer = Signer(secret_key=["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer(secret_key="my-secret-key", salt="custom-salt")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"custom-salt"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_sep():
    signer = Signer(secret_key="my-secret-key", sep="|")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer(secret_key="my-secret-key", key_derivation="concat")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.key_derivation == "concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_digest_method():
    from hashlib import sha256
    signer = Signer(secret_key="my-secret-key", digest_method=sha256)
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.digest_method == sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer(secret_key="my-secret-key", algorithm=algorithm)
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.algorithm == algorithm

def test_signer_constructor_with_invalid_separator():
    try:
        Signer(secret_key="my-secret-key", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == (
            "The given separator cannot be used because it may be"
            " contained in the signature itself. ASCII letters,"
            " digits, and '-_=' must not be used."
        )

def test_signer_constructor_with_none_salt():
    signer = Signer(secret_key="my-secret-key", salt=None)
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #17
#--------------------------

```python
def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) is True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"test-value"
    invalid_signature = b"invalid-signature"
    assert signer.verify_signature(value, invalid_signature) is False

def test_verify_signature_with_malformed_base64():
    signer = Signer("secret-key")
    value = b"test-value"
    malformed_signature = b"malformed-base64!!!"
    assert signer.verify_signature(value, malformed_signature) is False

def test_verify_signature_with_old_secret_key():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    old_signer = Signer("old-key")
    old_signature = old_signer.get_signature(value)
    assert signer.verify_signature(value, old_signature) is True

def test_verify_signature_with_new_secret_key():
    signer = Signer(["old-key", "new-key"])
    value = b"test-value"
    new_signature = signer.get_signature(value)
    assert signer.verify_signature(value, new_signature) is True

def test_verify_signature_with_wrong_secret_key():
    signer = Signer("secret-key")
    other_signer = Signer("other-secret-key")
    value = b"test-value"
    other_signature = other_signer.get_signature(value)
    assert signer.verify_signature(value, other_signature) is False


# LLM-generated content at query #18
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_iterable_secret_key():
    signer = Signer(["old_secret", "new_secret"])
    assert signer.secret_keys == [b"old_secret", b"new_secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt="custom_salt")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"custom_salt"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep="|")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"|"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is algorithm

def test_signer_constructor_with_none_salt():
    signer = Signer("secret", salt=None)
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_invalid_sep():
    import pytest
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        Signer("secret", sep="-")


