####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_credit_card_number_visa():
    from mimesis.providers.payment import Payment, CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card = payment.credit_card_number(card_type=CardType.VISA)
    
    assert card is not None
    assert isinstance(card, str)
    assert len(card) == 19  # 16 digits + 3 spaces
    assert card[0] == '4'
    parts = card.split()
    assert len(parts) == 4
    assert all(part.isdigit() for part in parts)
    assert all(len(part) == 4 for part in parts)


def test_credit_card_number_master_card():
    from mimesis.providers.payment import Payment, CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    
    assert card is not None
    assert isinstance(card, str)
    assert len(card) == 19  # 16 digits + 3 spaces
    parts = card.split()
    assert len(parts) == 4
    assert all(part.isdigit() for part in parts)
    assert all(len(part) == 4 for part in parts)
    first_two_digits = int(parts[0][:2])
    assert (2221 <= first_two_digits <= 2720) or (5100 <= first_two_digits <= 5599)


def test_credit_card_number_american_express():
    from mimesis.providers.payment import Payment, CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    
    assert card is not None
    assert isinstance(card, str)
    assert len(card) == 17  # 15 digits + 2 spaces
    parts = card.split()
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
    assert len(parts[0]) == 4
    assert len(parts[1]) == 6
    assert len(parts[2]) == 5
    first_digit = int(parts[0][0])
    assert first_digit in [3, 4]


def test_credit_card_number_default():
    from mimesis.providers.payment import Payment
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card = payment.credit_card_number()
    
    assert card is not None
    assert isinstance(card, str)
    assert len(card) in [17, 19]
    parts = card.split()
    assert all(part.isdigit() for part in parts)


def test_credit_card_number_none_card_type():
    from mimesis.providers.payment import Payment
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card = payment.credit_card_number(card_type=None)
    
    assert card is not None
    assert isinstance(card, str)
    assert len(card) in [17, 19]


def test_credit_card_number_format():
    from mimesis.providers.payment import Payment, CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card = payment.credit_card_number(card_type=CardType.VISA)
    
    parts = card.split()
    assert len(parts) == 4
    full_number = "".join(parts)
    assert len(full_number) == 16
    assert full_number.isdigit()


# LLM-generated content at query #2
#--------------------------

```python
def test_credit_card_number_visa():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    assert isinstance(card_number, str)
    assert len(card_number) == 19
    assert card_number[0] == '4'
    assert card_number[5] == ' '
    assert card_number[10] == ' '
    assert card_number[15] == ' '


def test_credit_card_number_master_card():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert isinstance(card_number, str)
    assert len(card_number) == 19
    assert card_number[5] == ' '
    assert card_number[10] == ' '
    assert card_number[15] == ' '


def test_credit_card_number_american_express():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert isinstance(card_number, str)
    assert len(card_number) == 17
    assert card_number[0] in ['3', '4']
    assert card_number[5] == ' '
    assert card_number[12] == ' '


def test_credit_card_number_default():
    from mimesis.providers.payment import Payment
    payment = Payment()
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number) in [17, 19]
    assert ' ' in card_number


def test_credit_card_number_format():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    parts = card_number.split(' ')
    assert len(parts) == 4
    assert all(part.isdigit() for part in parts)


def test_credit_card_number_american_express_format():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    parts = card_number.split(' ')
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_credit_card_number_deterministic():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment1 = Payment(seed=12345)
    payment2 = Payment(seed=12345)
    card1 = payment1.credit_card_number(CardType.VISA)
    card2 = payment2.credit_card_number(CardType.VISA)
    assert card1 == card2


# LLM-generated content at query #3
#--------------------------

```python
def test_credit_card_number_predicate_line_34_evaluates_to_false():
    """Test that the predicate at line 34 evaluates to False when str_num length >= length - 1."""
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    
    payment = Payment()
    
    # Generate a VISA card number (length = 16, so length - 1 = 15)
    # The initial number from randint(4000, 4999) is 4 digits
    # After converting to string, str_num will have 4 digits initially
    # The while loop condition is: while len(str_num) < length - 1
    # For VISA: while len(str_num) < 15
    # The loop will run until len(str_num) reaches 15
    # After the loop completes, len(str_num) should be 15
    # At this point, len(str_num) < length - 1 evaluates to 15 < 15 which is False
    
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    
    # The card_number should be formatted as "XXXX XXXX XXXX XXXX" (16 digits with spaces)
    # Removing spaces should give us 16 digits
    digits_only = card_number.replace(" ", "")
    assert len(digits_only) == 16
    
    # For AMERICAN_EXPRESS: length = 15, so length - 1 = 14
    card_number_amex = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    digits_only_amex = card_number_amex.replace(" ", "")
    assert len(digits_only_amex) == 15


# LLM-generated content at query #4
#--------------------------

```python
def test_credit_card_number_visa():
    from mimesis.providers.payment import Payment, CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(CardType.VISA)
    
    assert isinstance(card_number, str)
    assert len(card_number) == 19  # 16 digits + 3 spaces
    assert card_number[0] == '4'
    assert card_number[5] == ' '
    assert card_number[10] == ' '
    assert card_number[15] == ' '


def test_credit_card_number_mastercard():
    from mimesis.providers.payment import Payment, CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    
    assert isinstance(card_number, str)
    assert len(card_number) == 19  # 16 digits + 3 spaces
    first_digit = int(card_number[0])
    assert first_digit in [2, 5]
    assert card_number[5] == ' '
    assert card_number[10] == ' '
    assert card_number[15] == ' '


def test_credit_card_number_american_express():
    from mimesis.providers.payment import Payment, CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    
    assert isinstance(card_number, str)
    assert len(card_number) == 17  # 15 digits + 2 spaces
    assert card_number[0] in ['3', '4']
    assert card_number[4] == ' '
    assert card_number[11] == ' '


def test_credit_card_number_default():
    from mimesis.providers.payment import Payment
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number()
    
    assert isinstance(card_number, str)
    assert ' ' in card_number
    assert len(card_number) in [17, 19]


def test_credit_card_number_invalid_type():
    from mimesis.providers.payment import Payment
    from mimesis import Locale
    from mimesis.exceptions import NonEnumerableError
    
    payment = Payment(locale=Locale.EN)
    
    try:
        payment.credit_card_number("INVALID")
        assert False, "Expected an exception"
    except (NonEnumerableError, AttributeError, TypeError):
        assert True


def test_credit_card_number_format():
    from mimesis.providers.payment import Payment, CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(CardType.VISA)
    
    parts = card_number.split(' ')
    assert len(parts) == 4
    assert all(part.isdigit() for part in parts)
    assert all(len(part) == 4 for part in parts)


def test_credit_card_number_with_seed():
    from mimesis.providers.payment import Payment, CardType
    from mimesis import Locale
    
    payment1 = Payment(locale=Locale.EN, seed=12345)
    payment2 = Payment(locale=Locale.EN, seed=12345)
    
    card1 = payment1.credit_card_number(CardType.VISA)
    card2 = payment2.credit_card_number(CardType.VISA)
    
    assert card1 == card2


# LLM-generated content at query #5
#--------------------------

```python
def test_credit_card_number_predicate_line_34_evaluates_to_false():
    """Test that the predicate at line 34 evaluates to False when str_num length >= length - 1."""
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    
    payment = Payment()
    
    # Generate a credit card number with VISA (length = 16)
    # The predicate at line 34 is: while len(str_num) < length - 1
    # For VISA: length = 16, so the predicate should be: while len(str_num) < 15
    # After the while loop exits, len(str_num) should be >= 15
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    
    # Extract the digits only (remove spaces)
    digits_only = card_number.replace(" ", "")
    
    # For VISA, the final card number should be 16 digits
    # So len(str_num) before adding checksum should be 15
    # After the while loop exits, the predicate should be False
    assert len(digits_only) == 16
    assert len(digits_only) >= 15


# LLM-generated content at query #6
#--------------------------

```python
def test_credit_card_number_visa():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(CardType.VISA)
    
    assert isinstance(card_number, str)
    assert len(card_number) == 19
    assert card_number[0] == '4'
    assert card_number[5] == ' '
    assert card_number[10] == ' '
    assert card_number[15] == ' '


def test_credit_card_number_master_card():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    
    assert isinstance(card_number, str)
    assert len(card_number) == 19
    first_four = int(card_number[:4])
    assert (2221 <= first_four <= 2720) or (5100 <= first_four <= 5599)
    assert card_number[5] == ' '
    assert card_number[10] == ' '
    assert card_number[15] == ' '


def test_credit_card_number_american_express():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    
    assert isinstance(card_number, str)
    assert len(card_number) == 17
    assert card_number[0] in ['3', '4']
    assert card_number[4] == ' '
    assert card_number[11] == ' '


def test_credit_card_number_default():
    from mimesis.providers.payment import Payment
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number()
    
    assert isinstance(card_number, str)
    assert len(card_number) in [17, 19]
    assert ' ' in card_number


def test_credit_card_number_format():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(CardType.VISA)
    
    parts = card_number.split(' ')
    assert len(parts) == 4
    assert all(part.isdigit() for part in parts)
    assert all(len(part) == 4 for part in parts)


def test_credit_card_number_different_calls():
    from mimesis.providers.payment import Payment
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card1 = payment.credit_card_number()
    card2 = payment.credit_card_number()
    
    assert isinstance(card1, str)
    assert isinstance(card2, str)
    assert len(card1) > 0
    assert len(card2) > 0


# LLM-generated content at query #7
#--------------------------

```python
def test_credit_card_number_visa():
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    
    assert isinstance(card_number, str)
    assert len(card_number) == 19  # 16 digits + 3 spaces
    assert card_number[0] == '4'
    assert card_number.count(' ') == 3


def test_credit_card_number_mastercard():
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    
    assert isinstance(card_number, str)
    assert len(card_number) == 19  # 16 digits + 3 spaces
    assert card_number.count(' ') == 3
    first_four = card_number.replace(' ', '')[:4]
    assert first_four in ['2221', '2222', '2223', '2224', '2225', '2226', '2227', '2228', '2229', '2230', '2231', '2232', '2233', '2234', '2235', '2236', '2237', '2238', '2239', '2240', '2241', '2242', '2243', '2244', '2245', '2246', '2247', '2248', '2249', '2250', '2251', '2252', '2253', '2254', '2255', '2256', '2257', '2258', '2259', '2260', '2261', '2262', '2263', '2264', '2265', '2266', '2267', '2268', '2269', '2270', '2271', '2272', '2273', '2274', '2275', '2276', '2277', '2278', '2279', '2280', '2281', '2282', '2283', '2284', '2285', '2286', '2287', '2288', '2289', '2290', '2291', '2292', '2293', '2294', '2295', '2296', '2297', '2298', '2299', '2300', '2301', '2302', '2303', '2304', '2305', '2306', '2307', '2308', '2309', '2310', '2311', '2312', '2313', '2314', '2315', '2316', '2317', '2318', '2319', '2320', '2321', '2322', '2323', '2324', '2325', '2326', '2327', '2328', '2329', '2330', '2331', '2332', '2333', '2334', '2335', '2336', '2337', '2338', '2339', '2340', '2341', '2342', '2343', '2344', '2345', '2346', '2347', '2348', '2349', '2350', '2351', '2352', '2353', '2354', '2355', '2356', '2357', '2358', '2359', '2360', '2361', '2362', '2363', '2364', '2365', '2366', '2367', '2368', '2369', '2370', '2371', '2372', '2373', '2374', '2375', '2376', '2377', '2378', '2379', '2380', '2381', '2382', '2383', '2384', '2385', '2386', '2387', '2388', '2389', '2390', '2391', '2392', '2393', '2394', '2395', '2396', '2397', '2398', '2399', '2400', '2401', '2402', '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410', '2411', '2412', '2413', '2414', '2415', '2416', '2417', '2418', '2419', '2420', '2421', '2422', '2423', '2424', '2425', '2426', '2427', '2428', '2429', '2430', '2431', '2432', '2433', '2434', '2435', '2436', '2437', '2438', '2439', '2440', '2441', '2442', '2443', '2444', '2445', '2446', '2447', '2448', '2449', '2450', '2451', '2452', '2453', '2454', '2455', '2456', '2457', '2458', '2459', '2460', '2461', '2462', '2463', '2464', '2465', '2466', '2467', '2468', '2469', '2470', '2471', '2472', '2473', '2474', '2475', '2476', '2477', '2478', '2479', '2480', '2481', '2482', '2483', '2484', '2485', '2486', '2487', '2488', '2489', '2490', '2491', '2492', '2493', '2494', '2495', '2496', '2497', '2498', '2499', '2500', '2501', '2502', '2503', '2504', '2505', '2506', '2507', '2508', '2509', '2510', '2511', '2512', '2513', '2514', '2515', '2516', '2517', '2518', '2519', '2520', '2521', '2522', '2523', '2524', '2525', '2526', '2527', '2528', '2529', '2530', '2531', '2532', '2533', '2534', '2535', '2536', '2537', '2538', '2539', '2540', '2541', '2542', '2543', '2544', '2545', '2546', '2547', '2548', '2549', '2550', '2551', '2552', '2553', '2554', '2555', '2556', '2557', '2558', '2559', '2560', '2561', '2562', '2563', '2564', '2565', '2566', '2567', '2568', '2569', '2570', '2571', '2572', '2573', '2574', '2575', '2576', '2577', '2578', '2579', '2580', '2581', '2582', '2583', '2584', '2585', '2586', '2587', '2588', '2589', '2590', '2591', '2592', '2593', '2594', '2595', '2596', '2597', '2598', '2599', '2600', '2601', '2602', '2603', '2604', '2605', '2606', '2607', '2608', '2609', '2610', '2611', '2612', '2613', '2614', '2615', '2616', '2617', '2618', '2619', '2620', '2621', '2622', '2623', '2624', '2625', '2626', '2627', '2628', '2629', '2630', '2631', '2632', '2633', '2634', '2635', '2636', '2637', '2638', '2639', '2640', '2641', '2642', '2643', '2644', '2645', '2646', '2647', '2648', '2649', '2650', '2651', '2652', '2653', '2654', '2655', '2656', '2657', '2658', '2659', '2660', '2661', '2662', '2663', '2664', '2665', '2666


# LLM-generated content at query #8
#--------------------------

```python
def test_credit_card_number_predicate_line_34_false():
    """Test that the predicate at line 34 evaluates to False for VISA card."""
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN, seed=42)
    result = payment.credit_card_number(card_type=CardType.VISA)
    
    # For VISA: length = 16, so length - 1 = 15
    # The while loop condition is: len(str_num) < 15
    # After the loop exits, len(str_num) should be >= 15
    # We verify the result is a valid formatted credit card (16 digits with spaces)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_credit_card_number_visa():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    
    payment = Payment()
    card = payment.credit_card_number(CardType.VISA)
    
    assert isinstance(card, str)
    assert len(card) == 19
    assert card[0] == '4'
    assert card[5] == ' '
    assert card[10] == ' '
    assert card[15] == ' '


def test_credit_card_number_mastercard():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    
    payment = Payment()
    card = payment.credit_card_number(CardType.MASTER_CARD)
    
    assert isinstance(card, str)
    assert len(card) == 19
    first_four = int(card[:4].replace(' ', ''))
    assert (2221 <= first_four <= 2720) or (5100 <= first_four <= 5599)
    assert card[5] == ' '
    assert card[10] == ' '
    assert card[15] == ' '


def test_credit_card_number_american_express():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    
    payment = Payment()
    card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    
    assert isinstance(card, str)
    assert len(card) == 17
    assert card[0] in ['3']
    assert card[4] == ' '
    assert card[11] == ' '


def test_credit_card_number_default():
    from mimesis.providers.payment import Payment
    
    payment = Payment()
    card = payment.credit_card_number()
    
    assert isinstance(card, str)
    assert len(card) in [17, 19]
    assert ' ' in card


def test_credit_card_number_format():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    
    payment = Payment()
    card = payment.credit_card_number(CardType.VISA)
    
    parts = card.split(' ')
    assert len(parts) == 4
    for part in parts:
        assert part.isdigit()
        assert len(part) == 4


def test_credit_card_number_all_digits():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    
    payment = Payment()
    card = payment.credit_card_number(CardType.VISA)
    
    digits_only = card.replace(' ', '')
    assert digits_only.isdigit()


# LLM-generated content at query #2
#--------------------------

```python
def test_credit_card_number_predicate_at_line_34_evaluates_to_false():
    """Test that the predicate at line 34 evaluates to False when str_num length >= length - 1."""
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    
    payment = Payment()
    
    # For VISA card (length = 16), we need str_num to have length >= 15
    # VISA starts with 4000-4999 which gives us 4 digits
    # We need at least 15 digits total, so we need 11 more digits
    # The while loop condition is: while len(str_num) < length - 1
    # For VISA: while len(str_num) < 15
    # Once str_num has 15 or more digits, the condition is False
    
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    
    # The card_number should be formatted as "XXXX XXXX XXXX XXXX" (4 groups of 4 digits)
    # This means the actual number without spaces has 16 digits
    digits_only = card_number.replace(" ", "")
    
    # For VISA (length=16), the loop condition at line 34 should be False
    # when len(str_num) >= 15 (which is length - 1)
    assert len(digits_only) == 16
    assert card_number.count(" ") == 3
    
    # Test with AMERICAN_EXPRESS (length = 15)
    card_number_amex = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    digits_only_amex = card_number_amex.replace(" ", "")
    
    # For AMERICAN_EXPRESS (length=15), the loop condition at line 34 should be False
    # when len(str_num) >= 14 (which is length - 1)
    assert len(digits_only_amex) == 15
    assert card_number_amex.count(" ") == 2


# LLM-generated content at query #3
#--------------------------

```python
def test_credit_card_number_predicate_at_line_34_evaluates_to_false():
    """Test that the predicate at line 34 evaluates to False when str_num length >= length - 1."""
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    
    payment = Payment(seed=12345)
    
    # For VISA card (length = 16), we need str_num length >= 15
    # The initial number for VISA is between 4000-4999, which has 4 digits
    # So the predicate len(str_num) < length - 1 will be True initially
    # and the while loop will execute until str_num has at least 15 digits
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    
    # After the while loop completes, the predicate should evaluate to False
    # meaning the card number should be properly formatted
    assert card_number is not None
    assert isinstance(card_number, str)
    # VISA cards should have format: XXXX XXXX XXXX XXXX (16 digits with spaces)
    assert len(card_number.replace(" ", "")) == 16


# LLM-generated content at query #4
#--------------------------

```python
def test_credit_card_number_predicate_at_line_34_evaluates_to_false():
    """Test that the predicate at line 34 evaluates to False when str_num length is already >= length - 1."""
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    
    payment = Payment()
    
    # For VISA cards, length = 16, so we need len(str_num) >= 15
    # VISA starts with 4000-4999, which is 4 digits
    # We need to ensure that after str(number), len(str_num) is already >= length - 1
    # For AMERICAN_EXPRESS, length = 15, and it starts with 34 or 37 (2 digits)
    # So len(str_num) = 2, and length - 1 = 14, so the predicate would be True
    
    # For VISA, number is 4000-4999 (4 digits), length = 16, length - 1 = 15
    # So we need len(str_num) >= 15, which means str_num should have at least 15 digits
    # But str(4000-4999) is only 4 digits, so predicate is True
    
    # The predicate evaluates to False when len(str_num) >= length - 1
    # This happens after the while loop completes or never enters it
    # For AMERICAN_EXPRESS with 2-digit start and length=15, len(str_num)=2 < 14, enters loop
    # After loop fills to 14 digits, len(str_num) = 14, which is NOT < 14, so predicate is False
    
    card = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    
    # The predicate at line 34 should evaluate to False after the while loop
    # This means len(str_num) >= length - 1 at that point
    # For AMERICAN_EXPRESS, after padding, str_num should have exactly 14 digits
    # So len(str_num) = 14 and length - 1 = 14, making the predicate False
    
    assert card is not None
    assert isinstance(card, str)
    assert len(card) > 0


# LLM-generated content at query #5
#--------------------------

```python
def test_credit_card_number_visa():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    
    assert isinstance(card_number, str)
    assert len(card_number) == 19
    assert card_number[0] == '4'
    assert card_number[5] == ' '
    assert card_number[10] == ' '
    assert card_number[15] == ' '


def test_credit_card_number_mastercard():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    
    assert isinstance(card_number, str)
    assert len(card_number) == 19
    assert card_number[5] == ' '
    assert card_number[10] == ' '
    assert card_number[15] == ' '


def test_credit_card_number_american_express():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    
    assert isinstance(card_number, str)
    assert len(card_number) == 17
    assert card_number[0] in ['3', '4']
    assert card_number[4] == ' '
    assert card_number[11] == ' '


def test_credit_card_number_default():
    from mimesis.providers.payment import Payment
    
    payment = Payment()
    card_number = payment.credit_card_number()
    
    assert isinstance(card_number, str)
    assert len(card_number) in [17, 19]
    assert ' ' in card_number


def test_credit_card_number_format():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    
    parts = card_number.split(' ')
    assert len(parts) == 4
    for part in parts:
        assert part.isdigit()


def test_credit_card_number_invalid_type():
    from mimesis.providers.payment import Payment
    from mimesis.exceptions import NonEnumerableError
    
    payment = Payment()
    
    try:
        payment.credit_card_number("INVALID")
        assert False, "Expected exception"
    except (NonEnumerableError, AttributeError):
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_credit_card_number_predicate_line_34_false():
    """Test that the predicate at line 34 evaluates to False for AMERICAN_EXPRESS."""
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    
    payment = Payment()
    # AMERICAN_EXPRESS has length = 15, so length - 1 = 14
    # The initial number is either 34 or 37 (2 digits)
    # After str(number), str_num will have length >= 2
    # We need to ensure that after the while loop completes,
    # len(str_num) is NOT less than length - 1 (i.e., NOT less than 14)
    # This means the predicate at line 34 should evaluate to False
    
    card = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    
    # The card should be properly formatted and the predicate should have been False
    # when exiting the while loop, meaning str_num had length >= 14
    assert card is not None
    assert isinstance(card, str)
    # AMERICAN_EXPRESS format: 4 digits, 6 digits, 5 digits = "XXXX XXXXXX XXXXX"
    assert len(card.replace(" ", "")) == 15


# LLM-generated content at query #7
#--------------------------

```python
def test_credit_card_number_visa():
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")
    assert card_number.count(" ") == 3


def test_credit_card_number_master_card():
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.count(" ") == 3
    first_four = card_number.replace(" ", "")[:4]
    assert first_four[0] in ["2", "5"]


def test_credit_card_number_american_express():
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.count(" ") == 2
    assert card_number[0] in ["3"]


def test_credit_card_number_default():
    from mimesis.providers.payment import Payment
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number()
    
    assert isinstance(card_number, str)
    assert " " in card_number
    card_digits = card_number.replace(" ", "")
    assert len(card_digits) in [15, 16]
    assert card_digits[0] in ["3", "4", "5"]


def test_credit_card_number_none_card_type():
    from mimesis.providers.payment import Payment
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(card_type=None)
    
    assert isinstance(card_number, str)
    assert " " in card_number
    card_digits = card_number.replace(" ", "")
    assert len(card_digits) in [15, 16]


def test_credit_card_number_format():
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    
    parts = card_number.split(" ")
    assert len(parts) == 4
    for part in parts:
        assert part.isdigit()
        assert len(part) == 4


def test_credit_card_number_contains_only_digits_and_spaces():
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    from mimesis import Locale
    
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    
    for char in card_number:
        assert char.isdigit() or char == " "


def test_credit_card_number_reproducible_with_seed():
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    from mimesis import Locale
    
    payment1 = Payment(locale=Locale.EN, seed=42)
    card1 = payment1.credit_card_number(card_type=CardType.VISA)
    
    payment2 = Payment(locale=Locale.EN, seed=42)
    card2 = payment2.credit_card_number(card_type=CardType.VISA)
    
    assert card1 == card2


# LLM-generated content at query #8
#--------------------------

```python
def test_credit_card_number_predicate_at_line_34_evaluates_to_false():
    """Test that the predicate at line 34 evaluates to False when str_num length is sufficient."""
    from mimesis.providers.payment import Payment
    from mimesis.types import CardType
    
    payment = Payment()
    
    # Generate a credit card number with VISA (length should be 16)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    
    # The card_number should have spaces, remove them to check actual digit length
    card_digits = card_number.replace(" ", "")
    
    # For VISA, the final length should be 16 (with checksum), so the predicate at line 34
    # should have evaluated to False at some point (when len(str_num) >= length - 1, i.e., >= 15)
    assert len(card_digits) == 16
    assert card_digits.isdigit()


