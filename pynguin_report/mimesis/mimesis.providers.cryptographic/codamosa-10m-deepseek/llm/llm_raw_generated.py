####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method api_key of class Cryptographic
def test_Cryptographic_api_key():  # noqa: N802
    crypto = Cryptographic()
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 64  # 32 bytes in hex = 64 chars

    key = crypto.api_key(prefix='sk_')
    assert key.startswith('sk_')
    assert len(key) == 67  # 64 + 3

    key = crypto.api_key(fmt='base64')
    assert isinstance(key, str)
    assert len(key) == 43  # 32 bytes in base64 = 43 chars

    try:
        crypto.api_key(fmt='invalid')
        assert False, "Should raise ValueError"
    except ValueError:
        pass



# LLM-generated content at query #2
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint(): 
    crypto = Cryptographic()
    # Test with default algorithm (sha256)
    result = crypto.certificate_fingerprint()
    assert isinstance(result, str)
    assert len(result) == 95  # 32 bytes * 2 hex digits + 31 colons
    assert all(c in "0123456789ABCDEF:" for c in result)
    # Test with sha1 algorithm
    result = crypto.certificate_fingerprint(algorithm="sha1")
    assert isinstance(result, str)
    assert len(result) == 59  # 20 bytes * 2 hex digits + 19 colons
    assert all(c in "0123456789ABCDEF:" for c in result)
    # Test with unsupported algorithm
    try:
        crypto.certificate_fingerprint(algorithm="md5")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


# LLM-generated content at query #3
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint():


# LLM-generated content at query #4
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint(): 
    crypto = Cryptographic()
    result = crypto.certificate_fingerprint()
    assert isinstance(result, str)
    assert len(result) == 95  # 32 bytes * 2 hex digits + 31 colons
    assert all(c in '0123456789ABCDEF:' for c in result)
    assert result.count(':') == 31

    # Test with sha1
    result = crypto.certificate_fingerprint(algorithm='sha1')
    assert isinstance(result, str)
    assert len(result) == 59  # 20 bytes * 2 hex digits + 19 colons
    assert all(c in '0123456789ABCDEF:' for c in result)
    assert result.count(':') == 19

    # Test with invalid algorithm
    try:
        crypto.certificate_fingerprint(algorithm='md5')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


# LLM-generated content at query #5
#--------------------------

# Unit test for method api_key of class Cryptographic
def test_Cryptographic_api_key():  # noqa: N802
    # Test with default parameters
    crypto = Cryptographic()
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 64  # 32 bytes in hex = 64 characters

    # Test with prefix
    key = crypto.api_key(prefix='sk_')
    assert key.startswith('sk_')
    assert len(key) == 64 + 3  # 64 hex chars + 3 prefix chars

    # Test with base64 format
    key = crypto.api_key(fmt='base64')
    assert isinstance(key, str)
    # Base64 length can vary, but should be roughly length * 1.33
    assert 40 <= len(key) <= 50

    # Test with prefix and base64
    key = crypto.api_key(prefix='pk_', fmt='base64')
    assert key.startswith('pk_')

    # Test with custom length
    key = crypto.api_key(length=16)
    assert len(key) == 32  # 16 bytes in hex = 32 characters

    # Test invalid format
    try:
        crypto.api_key(fmt='invalid')
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown format" in str(e)


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint(): 
    # Test with default algorithm (sha256)
    crypto = Cryptographic()
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 95  # 32 bytes * 2 hex digits + 31 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)
    
    # Test with sha1 algorithm
    fingerprint = crypto.certificate_fingerprint(algorithm='sha1')
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 59  # 20 bytes * 2 hex digits + 19 colons
    assert all(c in "0123456789ABCDEF:" for c in fingerprint)
    
    # Test with invalid algorithm
    try:
        crypto.certificate_fingerprint(algorithm='invalid')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


# LLM-generated content at query #2
#--------------------------

# Unit test for method api_key of class Cryptographic
def test_Cryptographic_api_key():  # noqa: N802
    # Test with default parameters
    crypto = Cryptographic()
    key = crypto.api_key()
    assert isinstance(key, str)
    assert len(key) == 64  # 32 bytes in hex = 64 characters

    # Test with prefix
    key = crypto.api_key(prefix='sk_')
    assert key.startswith('sk_')
    assert len(key) == 67  # 64 + 3 for 'sk_'

    # Test with base64 format
    key = crypto.api_key(fmt='base64')
    assert isinstance(key, str)
    # Base64 length can vary, but should be at least length characters
    assert len(key) >= 32

    # Test with prefix and base64
    key = crypto.api_key(prefix='pk_', fmt='base64')
    assert key.startswith('pk_')

    # Test with custom length
    key = crypto.api_key(length=16)
    assert len(key) == 32  # 16 bytes in hex = 32 characters

    # Test with invalid format should raise ValueError
    try:
        crypto.api_key(fmt='invalid')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass



# LLM-generated content at query #3
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint(): # noqa: N802
    """Test method certificate_fingerprint of class Cryptographic."""
    crypto = Cryptographic()
    # Test with default algorithm (sha256)
    fingerprint = crypto.certificate_fingerprint()
    assert isinstance(fingerprint, str)
    # Should be colon-separated hex pairs
    parts = fingerprint.split(':')
    assert len(parts) == 32  # 256 bits = 32 bytes
    for part in parts:
        assert len(part) == 2
        int(part, 16)  # Should be valid hex

    # Test with sha1
    fingerprint = crypto.certificate_fingerprint(algorithm='sha1')
    assert isinstance(fingerprint, str)
    parts = fingerprint.split(':')
    assert len(parts) == 20  # 160 bits = 20 bytes
    for part in parts:
        assert len(part) == 2
        int(part, 16)

    # Test with invalid algorithm
    try:
        crypto.certificate_fingerprint(algorithm='invalid')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass



# LLM-generated content at query #4
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint():  # noqa: N802
    """Test method certificate_fingerprint of class Cryptographic."""
    crypto = Cryptographic()
    # Test with default algorithm (sha256)
    result = crypto.certificate_fingerprint()
    assert isinstance(result, str)
    # Check format: colon-separated hex pairs, uppercase
    parts = result.split(':')
    assert len(parts) == 32  # sha256 produces 32 bytes = 64 hex chars = 32 pairs
    for part in parts:
        assert len(part) == 2
        assert all(c in '0123456789ABCDEF' for c in part)
    # Test with sha1
    result = crypto.certificate_fingerprint(algorithm='sha1')
    assert isinstance(result, str)
    parts = result.split(':')
    assert len(parts) == 20  # sha1 produces 20 bytes = 40 hex chars = 20 pairs
    for part in parts:
        assert len(part) == 2
        assert all(c in '0123456789ABCDEF' for c in part)
    # Test with unsupported algorithm
    try:
        crypto.certificate_fingerprint(algorithm='md5')
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


# LLM-generated content at query #5
#--------------------------

# Unit test for method certificate_fingerprint of class Cryptographic
def test_Cryptographic_certificate_fingerprint():


