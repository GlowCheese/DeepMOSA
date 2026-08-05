####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_verify_signature_valid_data():
    signer = Signer(secret_key="test-secret")
    signature = signer.get_signature("test-value")
    assert signer.verify_signature("test-value", signature)

def test_verify_signature_invalid_signature():
    signer = Signer(secret_key="test-secret")
    assert not signer.verify_signature("test-value", b"invalid-signature")

def test_verify_signature_with_bytes():
    signer = Signer(secret_key=b"test-secret")
    signature = signer.get_signature(b"test-value")
    assert signer.verify_signature(b"test-value", signature)

def test_verify_signature_with_different_secret_key():
    signer = Signer(secret_key="secret1")
    signature = signer.get_signature("value")
    assert not signer.verify_signature("value", signature)

def test_verify_signature_with_key_rotation():
    signer = Signer(secret_key=["old-secret", "new-secret"])
    old_signer = Signer(secret_key="old-secret")
    signature = old_signer.get_signature("value")
    assert signer.verify_signature("value", signature)

def test_verify_signature_with_invalid_base64():
    signer = Signer(secret_key="test-secret")
    assert not signer.verify_signature("value", b"!!!invalid-base64!!!")

def test_verify_signature_with_empty_value():
    signer = Signer(secret_key="test-secret")
    signature = signer.get_signature("")
    assert signer.verify_signature("", signature)

def test_verify_signature_with_none_salt():
    signer = Signer(secret_key="test-secret", salt=None)
    signature = signer.get_signature("value")
    assert signer.verify_signature("value", signature)

def test_verify_signature_with_custom_sep():
    signer = Signer(secret_key="test-secret", sep=b"|")
    signature = signer.get_signature("value")
    assert signer.verify_signature("value", signature)
```


# LLM-generated content at query #2
#--------------------------

def test_derive_key_default_secret_key():
    signer = Signer(b"secret")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) > 0

def test_derive_key_concat():
    signer = Signer(b"secret", key_derivation="concat")
    key = signer.derive_key()
    assert isinstance(key, bytes)

def test_derive_key_django_concat():
    signer = Signer(b"secret", key_derivation="django-concat")
    key = signer.derive_key()
    assert isinstance(key, bytes)

def test_derive_key_hmac():
    signer = Signer(b"secret", key_derivation="hmac")
    key = signer.derive_key()
    assert isinstance(key, bytes)

def test_derive_key_none():
    signer = Signer(b"secret", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret"

def test_derive_key_with_explicit_secret():
    signer = Signer(b"other")
    key = signer.derive_key(b"custom")
    assert isinstance(key, bytes)

def test_derive_key_with_unknown_method_raises():
    signer = Signer(b"secret", key_derivation="unknown")
    try:
        signer.derive_key()
        assert False
    except TypeError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_verify_signature_valid_signature():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_invalid_signature():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b"test-value"
    assert signer.verify_signature(value, b"invalid-signature") == False

def test_verify_signature_with_bytes_value():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b"bytes-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_string_value():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = "string-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_key_rotation():
    signer = Signer(["old-key", "new-key"], salt="salt", sep=".")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_old_key():
    signer = Signer(["old-key", "new-key"], salt="salt", sep=".")
    value = b"test-value"
    old_signer = Signer("old-key", salt="salt", sep=".")
    old_signature = old_signer.get_signature(value)
    assert signer.verify_signature(value, old_signature) == True

def test_verify_signature_with_invalid_base64():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b"test-value"
    assert signer.verify_signature(value, b"!!!invalid-base64!!!") == False

def test_verify_signature_with_empty_value():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b""
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_different_separator():
    signer = Signer("secret-key", salt="salt", sep="-")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_none_secret_key():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b"test-value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True
```


# LLM-generated content at query #4
#--------------------------

def test_unsign_valid_signature():
    signer = Signer("secret-key", salt="salt")
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"

def test_unsign_no_separator():
    signer = Signer("secret-key", salt="salt")
    try:
        signer.unsign(b"test_value")
        assert False
    except BadSignature:
        pass

def test_unsign_wrong_signature():
    signer = Signer("secret-key", salt="salt")
    signed = b"test_value." + b"wrong_signature"
    try:
        signer.unsign(signed)
        assert False
    except BadSignature:
        pass

def test_unsign_empty_value():
    signer = Signer("secret-key", salt="salt")
    signed = signer.sign(b"")
    result = signer.unsign(signed)
    assert result == b""

def test_unsign_with_different_sep():
    signer = Signer("secret-key", salt="salt", sep=b":")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_unicode_value():
    signer = Signer("secret-key", salt="salt")
    signed = signer.sign("héllo")
    result = signer.unsign(signed)
    assert result == "héllo".encode("utf-8")


# LLM-generated content at query #5
#--------------------------

def test_verify_signature_invalid_base64_returns_false():
    signer = Signer("secret-key")
    result = signer.verify_signature(b"test", "invalid-base64!!!")
    assert result == False


# LLM-generated content at query #6
#--------------------------

def test_verify_signature_base64_decode_exception_returns_false():
    signer = Signer("secret-key")
    result = signer.verify_signature(b"test_value", b"!!!invalid-base64!!!")
    assert result == False


# LLM-generated content at query #7
#--------------------------

def test_signer_constructor_with_str_secret_key():
    signer = Signer(secret_key="my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(secret_key=b"my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]

def test_signer_constructor_with_list_of_str_secret_keys():
    signer = Signer(secret_key=["key1", "key2", "key3"])
    assert signer.secret_keys == [b"key1", b"key2", b"key3"]

def test_signer_constructor_with_list_of_bytes_secret_keys():
    signer = Signer(secret_key=[b"key1", b"key2", b"key3"])
    assert signer.secret_keys == [b"key1", b"key2", b"key3"]

def test_signer_constructor_with_custom_salt():
    signer = Signer(secret_key="key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_bytes_salt():
    signer = Signer(secret_key="key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_none_salt():
    signer = Signer(secret_key="key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_sep():
    signer = Signer(secret_key="key", sep=b"|")
    assert signer.sep == b"|"

def test_signer_constructor_with_sep_in_base64_alphabet_raises():
    try:
        Signer(secret_key="key", sep=b"a")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer(secret_key="key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method():
    signer = Signer(secret_key="key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = Signer(secret_key="key", algorithm=algorithm)
    assert signer.algorithm == algorithm


# LLM-generated content at query #8
#--------------------------

def test_signer_constructor_defaults():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.digest_method, type)
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_list_of_strings():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_list_of_bytes():
    signer = Signer([b"key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_none_salt():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep="|")
    assert signer.sep == b"|"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algo = HMACAlgorithm()
    signer = Signer("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_signer_constructor_raises_on_sep_in_base64():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret", sep="a")
    with pytest.raises(ValueError):
        Signer("secret", sep="-")
    with pytest.raises(ValueError):
        Signer("secret", sep="_")
    with pytest.raises(ValueError):
        Signer("secret", sep="=")


# LLM-generated content at query #9
#--------------------------

def test_signer_constructor_default_parameters():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_all_parameters():
    signer = Signer(
        secret_key=["key1", "key2"],
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.sep == b"|"
    assert signer.salt == b"custom-salt"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_separator_in_base64_alphabet_raises():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret", sep=b"A")

def test_signer_constructor_secret_key_as_bytes():
    signer = Signer(b"bytes-key")
    assert signer.secret_keys == [b"bytes-key"]

def test_signer_constructor_secret_key_as_list():
    signer = Signer(["key1", b"key2", "key3"])
    assert signer.secret_keys == [b"key1", b"key2", b"key3"]

def test_signer_constructor_salt_none():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #10
#--------------------------

def test_signer_constructor_default_parameters():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_separator():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_salt():
    signer = Signer("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_digest_method():
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_algorithm():
    algo = HMACAlgorithm(hashlib.sha256)
    signer = Signer("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_signer_constructor_with_separator_in_base64_alphabet():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret", sep=b".")

def test_signer_constructor_with_empty_separator():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret", sep=b"")

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_list_of_secret_keys():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_none_salt():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_hmac_algorithm():
    signer = Signer("secret", algorithm=HMACAlgorithm())
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #11
#--------------------------

def test_signer_constructor_defaults():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_list_of_strings():
    signer = Signer(["old", "new"])
    assert signer.secret_keys == [b"old", b"new"]

def test_signer_constructor_with_list_of_bytes():
    signer = Signer([b"old", b"new"])
    assert signer.secret_keys == [b"old", b"new"]

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt=b"custom")
    assert signer.salt == b"custom"

def test_signer_constructor_with_custom_salt_string():
    signer = Signer("secret", salt="custom")
    assert signer.salt == b"custom"

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_sep_string():
    signer = Signer("secret", sep=":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_invalid_sep():
    import string
    for char in string.ascii_letters + string.digits + "-_=":
        try:
            Signer("secret", sep=char)
            assert False, f"Should raise ValueError for sep '{char}'"
        except ValueError:
            pass

def test_signer_constructor_salt_none():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #12
#--------------------------

def test_verify_signature_base64_decode_raises_exception():
    signer = Signer("test-secret-key")
    result = signer.verify_signature(b"test-value", b"invalid-base64!!!")
    assert result is False


# LLM-generated content at query #13
#--------------------------

def test_verify_signature_with_valid_signature():
    signer = Signer(secret_key="secret")
    value = b"test"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_invalid_signature():
    signer = Signer(secret_key="secret")
    assert signer.verify_signature(b"test", b"invalidsig") == False

def test_verify_signature_with_empty_value():
    signer = Signer(secret_key="secret")
    sig = signer.get_signature(b"")
    assert signer.verify_signature(b"", sig) == True

def test_verify_signature_with_invalid_base64_sig():
    signer = Signer(secret_key="secret")
    assert signer.verify_signature(b"test", b"!!!invalidbase64!!!") == False

def test_verify_signature_with_different_secret_key():
    signer1 = Signer(secret_key="secret1")
    signer2 = Signer(secret_key="secret2")
    sig = signer1.get_signature(b"test")
    assert signer2.verify_signature(b"test", sig) == False

def test_verify_signature_with_key_rotation():
    signer = Signer(secret_key=["old_secret", "new_secret"])
    value = b"test"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_old_key_signature():
    signer = Signer(secret_key=["old_secret", "new_secret"])
    value = b"test"
    old_signer = Signer(secret_key="old_secret")
    sig = old_signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_none_secret_key():
    signer = Signer(secret_key="secret", key_derivation="none")
    value = b"test"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_hmac_derivation():
    signer = Signer(secret_key="secret", key_derivation="hmac")
    value = b"test"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_concat_derivation():
    signer = Signer(secret_key="secret", key_derivation="concat")
    value = b"test"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_django_concat_derivation():
    signer = Signer(secret_key="secret", key_derivation="django-concat")
    value = b"test"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True


# LLM-generated content at query #14
#--------------------------

def test_signer_constructor_defaults():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret():
    signer = Signer(b"secret_key")
    assert signer.secret_keys == [b"secret_key"]

def test_signer_constructor_with_list_of_strings():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_list_of_bytes():
    signer = Signer([b"key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_sep_str():
    signer = Signer("secret", sep=":")
    assert signer.sep == b":"

def test_signer_constructor_with_none_salt():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_custom_salt_str():
    signer = Signer("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_key_derivation_concat():
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_key_derivation_hmac():
    signer = Signer("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_key_derivation_none():
    signer = Signer("secret", key_derivation="none")
    assert signer.key_derivation == "none"

def test_signer_constructor_with_custom_digest_method():
    custom_digest = lambda: None
    signer = Signer("secret", digest_method=custom_digest)
    assert signer.digest_method == custom_digest

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_with_sep_in_base64_alphabet_raises():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret", sep=b"a")


# LLM-generated content at query #15
#--------------------------

def test_verify_signature_returns_false_when_base64_decode_raises_exception():
    signer = Signer("test-secret-key")
    result = signer.verify_signature(b"test-value", b"invalid!base64")
    assert result is False


# LLM-generated content at query #16
#--------------------------

def test_signer_constructor_defaults():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_list_of_secret_keys():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep=b"|")
    assert signer.sep == b"|"

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_none_salt():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algo = HMACAlgorithm()
    signer = Signer("secret", algorithm=algo)
    assert signer.algorithm is algo


# LLM-generated content at query #17
#--------------------------

def test_verify_signature_returns_false_when_base64_decode_raises_exception():
    signer = Signer(secret_key="test-secret")
    result = signer.verify_signature("test-value", "invalid-base64!!!")
    assert result is False


# LLM-generated content at query #18
#--------------------------

def test_signer_constructor_default_parameters():
    signer = Signer(secret_key="secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(secret_key=b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_list_of_secret_keys():
    signer = Signer(secret_key=["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_separator():
    signer = Signer(secret_key="secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt():
    signer = Signer(secret_key="secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_key_derivation_concat():
    signer = Signer(secret_key="secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_key_derivation_hmac():
    signer = Signer(secret_key="secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer(secret_key="secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer(secret_key="secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_separator_not_in_base64_alphabet():
    signer = Signer(secret_key="secret", sep=b"!")
    assert signer.sep == b"!"


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_verify_signature_with_valid_signature_returns_true():
    signer = Signer(secret_key="secret-key", salt="salt", sep=".")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_invalid_signature_returns_false():
    signer = Signer(secret_key="secret-key", salt="salt", sep=".")
    value = b"test-value"
    sig = b"invalid-signature"
    assert signer.verify_signature(value, sig) == False

def test_verify_signature_with_different_secret_key_returns_false():
    signer1 = Signer(secret_key="secret-key-1", salt="salt", sep=".")
    signer2 = Signer(secret_key="secret-key-2", salt="salt", sep=".")
    value = b"test-value"
    sig = signer1.get_signature(value)
    assert signer2.verify_signature(value, sig) == False

def test_verify_signature_with_multiple_secret_keys_validates_old_key():
    signer = Signer(secret_key=["old-key", "new-key"], salt="salt", sep=".")
    value = b"test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_invalid_base64_signature_returns_false():
    signer = Signer(secret_key="secret-key", salt="salt", sep=".")
    value = b"test-value"
    sig = b"!!!invalid-base64!!!"
    assert signer.verify_signature(value, sig) == False

def test_verify_signature_with_empty_value_returns_true():
    signer = Signer(secret_key="secret-key", salt="salt", sep=".")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_string_inputs_returns_true():
    signer = Signer(secret_key="secret-key", salt="salt", sep=".")
    value = "test-value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True
```


# LLM-generated content at query #2
#--------------------------

def test_derive_key_default_secret_key():
    signer = Signer("secret")
    key = signer.derive_key()
    assert isinstance(key, bytes)

def test_derive_key_concat():
    signer = Signer("secret", key_derivation="concat")
    key = signer.derive_key()
    assert isinstance(key, bytes)

def test_derive_key_django_concat():
    signer = Signer("secret", key_derivation="django-concat")
    key = signer.derive_key()
    assert isinstance(key, bytes)

def test_derive_key_hmac():
    signer = Signer("secret", key_derivation="hmac")
    key = signer.derive_key()
    assert isinstance(key, bytes)

def test_derive_key_none():
    signer = Signer("secret", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret"

def test_derive_key_with_specific_secret():
    signer = Signer("secret")
    key = signer.derive_key(secret_key="other")
    assert isinstance(key, bytes)

def test_derive_key_unknown_method():
    signer = Signer("secret", key_derivation="unknown")
    try:
        signer.derive_key()
        assert False
    except TypeError:
        pass

def test_derive_key_with_salt():
    signer = Signer("secret", salt="custom_salt")
    key = signer.derive_key()
    assert isinstance(key, bytes)


# LLM-generated content at query #3
#--------------------------

```python
def test_signer_constructor_default_parameters():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"bytes-secret")
    assert signer.secret_keys == [b"bytes-secret"]

def test_signer_constructor_with_list_of_keys():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_separator():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algo = HMACAlgorithm()
    signer = Signer("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_signer_constructor_separator_in_base64_alphabet_raises():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret", sep=b"a")
```


# LLM-generated content at query #4
#--------------------------

def test_signer_constructor_default_parameters():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_list_of_strings():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_list_of_bytes():
    signer = Signer([b"key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_custom_digest_method():
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = Signer("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_sep_in_base64_alphabet_raises():
    import re
    from itsdangerous.signer import _base64_alphabet
    try:
        signer = Signer("secret", sep="a")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #5
#--------------------------

def test_verify_signature_returns_false_on_invalid_base64_sig():
    signer = Signer(secret_key=b"secret-key")
    result = signer.verify_signature(b"test", b"not-base64!!!")
    assert result is False


# LLM-generated content at query #6
#--------------------------

def test_verify_signature_returns_false_when_sig_is_invalid_base64():
    signer = Signer(secret_key="test-secret")
    value = b"test-value"
    invalid_sig = "!!!invalid-base64!!!"
    result = signer.verify_signature(value, invalid_sig)
    assert result == False


# LLM-generated content at query #7
#--------------------------

def test_signer_constructor_defaults():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.digest_method, staticmethod)
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_signer_constructor_with_list():
    signer = Signer(["key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_sep():
    signer = Signer("secret-key", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_salt():
    signer = Signer("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_salt_none():
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_key_derivation():
    signer = Signer("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_digest_method():
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.algorithm == algorithm

def test_signer_constructor_with_sep_in_base64_alphabet():
    import string
    for char in string.ascii_letters + string.digits + "-_=":
        try:
            Signer("secret-key", sep=char.encode())
        except ValueError:
            pass
        else:
            assert False, f"Expected ValueError for sep {char!r}"


# LLM-generated content at query #8
#--------------------------

def test_verify_signature_with_invalid_base64_returns_false():
    signer = Signer(secret_key=b"secret")
    result = signer.verify_signature(b"test", b"!!!invalid-base64!!!")
    assert result is False


# LLM-generated content at query #9
#--------------------------

def test_signer_constructor_defaults():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_list_of_secrets():
    signer = Signer(["old_secret", "new_secret"])
    assert signer.secret_keys == [b"old_secret", b"new_secret"]

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_with_none_salt():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_invalid_separator():
    import string
    for char in string.ascii_letters + string.digits + "-_=":
        try:
            Signer("secret", sep=char.encode())
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for separator {char!r}")


# LLM-generated content at query #10
#--------------------------

def test_signer_constructor_default_parameters():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"bytes-key")
    assert signer.secret_keys == [b"bytes-key"]

def test_signer_constructor_with_list_of_secret_keys():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_separator():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_salt_none():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_separator_in_base64_raises_error():
    try:
        Signer("secret", sep=b"+")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #11
#--------------------------

def test_verify_signature_valid_signature():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_invalid_signature():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    assert signer.verify_signature(value, b"invalid_sig") == False

def test_verify_signature_with_bytes_value():
    signer = Signer("secret-key", salt="salt")
    value = b"bytes value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_string_value():
    signer = Signer("secret-key", salt="salt")
    value = "string value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_multiple_secret_keys():
    signer = Signer(["old_key", "new_key"], salt="salt")
    value = b"test"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_old_secret_key():
    signer = Signer(["old_key", "new_key"], salt="salt")
    value = b"test"
    old_signer = Signer("old_key", salt="salt")
    old_sig = old_signer.get_signature(value)
    assert signer.verify_signature(value, old_sig) == True

def test_verify_signature_returns_false_for_invalid_base64():
    signer = Signer("secret-key")
    assert signer.verify_signature(b"value", b"!!!invalid_base64!!!") == False

def test_verify_signature_returns_false_for_wrong_key():
    signer1 = Signer("key1")
    signer2 = Signer("key2")
    value = b"test"
    sig = signer1.get_signature(value)
    assert signer2.verify_signature(value, sig) == False


# LLM-generated content at query #12
#--------------------------

def test_verify_signature_invalid_base64_returns_false():
    signer = Signer("test-secret")
    result = signer.verify_signature(b"test-value", b"!!!invalid-base64!!!")
    assert result is False


# LLM-generated content at query #13
#--------------------------

def test_signer_constructor_default_parameters() -> None:
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_signer_constructor_with_bytes_secret_key() -> None:
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_signer_constructor_with_list_of_strings() -> None:
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_list_of_bytes() -> None:
    signer = Signer([b"key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_salt() -> None:
    signer = Signer("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_custom_separator() -> None:
    signer = Signer("secret-key", sep=b"|")
    assert signer.sep == b"|"

def test_signer_constructor_with_custom_key_derivation() -> None:
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method() -> None:
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_signer_constructor_with_custom_algorithm() -> None:
    algorithm = HMACAlgorithm()
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_separator_in_base64_alphabet_raises() -> None:
    try:
        Signer("secret-key", sep=b".")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for '.' separator"


# LLM-generated content at query #14
#--------------------------

```python
def test_signer_constructor_with_string_secret_key():
    signer = Signer("my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]

def test_signer_constructor_with_list_of_strings():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_list_of_bytes():
    signer = Signer([b"key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_sep():
    signer = Signer("key", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt():
    signer = Signer("key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_none_salt():
    signer = Signer("key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_secret_key_property():
    signer = Signer(["old_key", "new_key"])
    assert signer.secret_key == b"new_key"

def test_signer_constructor_raises_error_for_invalid_sep():
    import pytest
    with pytest.raises(ValueError):
        Signer("key", sep=b"+")
```


# LLM-generated content at query #15
#--------------------------

def test_verify_signature_valid_signature():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_invalid_signature():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    assert signer.verify_signature(value, b"invalid_sig") == False

def test_verify_signature_different_key():
    signer1 = Signer("secret-key-1", salt="salt")
    signer2 = Signer("secret-key-2", salt="salt")
    value = b"test value"
    sig = signer1.get_signature(value)
    assert signer2.verify_signature(value, sig) == False

def test_verify_signature_with_key_rotation():
    signer = Signer(["old-key", "new-key"], salt="salt")
    value = b"test value"
    old_sig = Signer(["old-key"], salt="salt").get_signature(value)
    new_sig = signer.get_signature(value)
    assert signer.verify_signature(value, old_sig) == True
    assert signer.verify_signature(value, new_sig) == True

def test_verify_signature_bytes_value():
    signer = Signer("secret-key", salt="salt")
    value = b"test bytes value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_string_value():
    signer = Signer("secret-key", salt="salt")
    value = "test string value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_empty_value():
    signer = Signer("secret-key", salt="salt")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_invalid_base64_sig():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    assert signer.verify_signature(value, b"!!!invalid_base64!!!") == False

def test_verify_signature_none_sig():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    assert signer.verify_signature(value, b"") == False


# LLM-generated content at query #16
#--------------------------

def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"test_value"
    invalid_sig = b"invalid_signature"
    assert signer.verify_signature(value, invalid_sig) == False

def test_verify_signature_with_different_key():
    signer = Signer("secret-key")
    value = b"test_value"
    sig = signer.get_signature(value)
    signer2 = Signer("different-secret")
    assert signer2.verify_signature(value, sig) == False

def test_verify_signature_with_key_rotation():
    signer = Signer(["old-key", "new-key"])
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_old_key():
    signer = Signer(["old-key", "new-key"])
    value = b"test_value"
    old_signer = Signer("old-key")
    old_sig = old_signer.get_signature(value)
    assert signer.verify_signature(value, old_sig) == True

def test_verify_signature_with_base64_decode_exception():
    signer = Signer("secret-key")
    value = b"test_value"
    invalid_sig = b"!!!invalid_base64!!"
    assert signer.verify_signature(value, invalid_sig) == False

def test_verify_signature_with_string_value():
    signer = Signer("secret-key")
    value = "test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_empty_value():
    signer = Signer("secret-key")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_unicode_value():
    signer = Signer("secret-key")
    value = "héllo"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True


