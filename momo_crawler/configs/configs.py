HEADER = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1700.107 Safari/537.36',
}

HEADER2 = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36',
}

COOKIES = {
    '_ts_id': '999999999999999999',
}

COOKIES2 = {
    '_ts_id': '888888888888888888',
}

RESULT_FEATURE_LIST = ('GID', 'payment_credit_card', \
            'payment_arrival', 'payment_convenience_store', 'payment_ATM', 'payment_iBon', \
            'img_height', 'is_warm', 'is_cold', 'is_bright', 'is_dark', \
            '12H', 'shopcart',\
            'have_video', 'origin', \
            'supplementary', 'bottle', 'combination', 'look_times', 'label')

DETERGENT_FEATURE_LIST = ['GID', 'unitPrice', 'price', 'volume', 'img_height', 'is_warm',\
                           'is_cold', 'is_bright', 'is_dark', '12H', 'superstore', 'haveVideo',\
                           'haveOrigin', 'supplementary', 'bottle', 'combination', 'look_times',\
                           'label']

BODYWASH_FEATURE_LIST = ['GID', 'unitPrice', 'haveOrigin', 'volume', 'supplementary', 'bottle',\
                           'combination', 'price', 'payment_ConvenienceStore', 'img_height',\
                           'is_warm', 'is_cold', '12H', 'haveVideo', 'installments', 'look_times',\
                           'label']

ESSENCE_FEATURE_LIST = ['GID', 'label', 'price', 'haveOrigin', 'unitPrice', 'volume',\
                           'discount', 'img_height', 'is_warm', 'is_cold', 'is_bright', 'is_dark',\
                           '12H', 'superstore', 'haveVideo', 'brand', 'installment', 'wrinkle',\
                           'whitening', 'moist', 'allergy', 'pimples', 'sunscreen', 'look_times']