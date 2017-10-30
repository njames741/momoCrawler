from utils.utils import *

def get_detergent_features(soup, goods_icode):
    """
    ['GID', 'unitPrice', 'price', 'volume', 'img_height', 'is_warm',
     'is_cold', 'is_bright', 'is_dark', '12H', 'superstore', 'haveVideo',
     'haveOrigin', 'supplementary', 'bottle', 'combination', 'look_times',
     'label']
    """
    row_list = list()
    row_list.append(goods_icode)
    row_list.append(price(soup))
    row_list.append(discount(soup))
    row_list += payment(soup)
    row_list.append(preferentialCount(soup))
    img_result_list = image_analysis(soup)
    row_list.append(img_result_list[0])
    row_list += img_result_list[1]
    row_list += img_result_list[2]
    row_list += transport(soup)
    row_list.append(productFormatCount(soup))
    row_list.append(attributesListArea(soup))
    row_list.append(haveVideo(soup))
    row_list += origin(soup)
    row_list += unit(soup)
    row_list.append(look_num)
    row_list.append(label)
    return row_list

def get_bodywash_features():
    """
    ['GID', 'unitPrice', 'haveOrigin', 'volume', 'supplementary', 'bottle',
     'combination', 'price', 'payment_ConvenienceStore', 'img_height',
     'is_warm', 'is_cold', '12H', 'haveVideo', 'installments', 'look_times',
     'label']
    """
    pass

def get_essense_features():
    """
    ['GID', 'label', 'price', 'haveOrigin', 'unitPrice', 'volume',
     'discount', 'img_height', 'is_warm', 'is_cold', 'is_bright', 'is_dark',
     '12H', 'superstore', 'haveVideo', 'brand', 'installment', 'wrinkle',
     'whitening', 'moist', 'allergy', 'pimples', 'sunscreen', 'look_times']
    """
    pass
