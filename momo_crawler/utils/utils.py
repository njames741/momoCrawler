"""
utils module 進行所有特徵萃取的工作
"""
from bs4 import BeautifulSoup
import urllib.request, urllib.error, urllib.parse
import jieba
import numpy as np
import requests
import re
import io
import sys
import time
import traceback
from PIL import Image
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from configs.configs import HEADER, COOKIES

jieba.set_dictionary('dict.txt.big')

# # 價錢
# def price(soup):
# 	try:
# 		price = soup.find('li','special').find('span').text
# 	except:
# 		price = soup.find('ul' ,'prdPrice').find('li').find('del').text
# 	price = price.replace(",","")
# 	# print "price: ",int(price)
# 	return int(price)

# # 折扣
# def discount(soup):
# 	try:
# 		OldPrice = soup.find('ul' ,'prdPrice').find('li').find('del').text.replace(",","")
# 		NewPrice = soup.find('li','special').find('span').text.replace(",","")
# 		return (int(OldPrice) - int(NewPrice))
# 	except:
# 		return 0

"""
付款方式(one hot encoding)
會產生5個columns: 'payment_credit_card', 'payment_arrival', 'payment_convenience_store', 'payment_ATM', 'payment_iBon'
"""
def payment(soup):
	paymentList = ['信用卡','貨到付款', '超商付款', 'ATM', 'iBon']
	paymentFeature = list()
	payment = soup.find('dl','payment').text.split("\n")
	for i in paymentList:
		if i in payment:
			paymentFeature.append(1)
		else:
			paymentFeature.append(0)
	return paymentFeature

# # 贈品(數量)
# def preferentialCount(soup):
# 	try:
# 		preferential = soup.find('dl','preferential').findAll('dd')
# 		return len(preferential)
# 	except:
# 		return 0

# # 庫存倒數
# def reciprocal(soup):
# 	reciprocal = soup.select('#goodsDtCount_001')[0]['value']
# 	if int(reciprocal) <= 5:
# 		return 1
# 	else:
# 		return 0

"""
圖片分析
會產生5個columns : 'img_height', 'is_warm', 'is_cold', 'is_bright', 'is_dark'
"""
def image_analysis(soup):
	# vendordetailview 是整個「商品特色」頁面的標籤
	vendordetailview = soup.find('div', class_='vendordetailview')
	iframe = vendordetailview.find('iframe')
	iframesrc = iframe['src']
	iframe_web = 'https://www.momoshop.com.tw' + iframesrc
	iframe_requests = requests.get(iframe_web, headers=HEADER, cookies=COOKIES)
	iframe_soup = BeautifulSoup(iframe_requests.text, 'html.parser')

	opener = urllib.request.build_opener()
	opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
	imgs = iframe_soup.find_all('img')

	if len(imgs) == 0:
		return 0, [0, 0], [0, 0]

	height_sum = 0
	r_sum, b_sum = 0, 0
	brightness_sum = 0

	for img in imgs:
		imgsrc = img['src'].split('?')[0]
		if imgsrc[:6] == '/exper':
			imgsrc = 'https://www.momoshop.com.tw' + imgsrc
		if imgsrc[:5] == '//img':
			imgsrc = 'https:' + imgsrc

		if imgsrc[8:12] == 'img1' or imgsrc[8:12] == 'img2':
			imgsrc = 'https://img3' + imgsrc[12:]
		imgsrc = imgsrc.replace('"','')
		print(imgsrc + '  <======== 去爬的圖片網址')
		
		try:
			image_file = opener.open(imgsrc)
			print('成功的', imgsrc)
		except Exception as e:
			print('==============img url錯誤====================')
			traceback.print_exc()
			print('失敗的', imgsrc)
		time.sleep(1)
		temp_image = io.BytesIO(urllib.request.urlopen(imgsrc).read())
		image = Image.open(temp_image)

		# 處理色溫
		rgb_sum_list = _color_temp(image)
		r_sum += rgb_sum_list[0]
		b_sum += rgb_sum_list[2]

		# 處理高度
		width, height = image.size
		height_sum += height

		# 處理亮度
		brightness_sum += _get_brightness(image)

	# 色溫
	if r_sum > b_sum: temp_list = [1, 0]
	else: temp_list = [0, 1]

	# 平均亮度計算
	brightness_avg = brightness_sum / len(imgs)
	if brightness_avg > 127: brightness_list = [1, 0]
	else: brightness_list = [0, 1]

	return height_sum, temp_list, brightness_list

def _get_brightness(img):
	width, height = img.size
	pixels = img.load()
	pixels_avg_list = list()
	for w in range(width):
		for h in range(height):
			pixels_avg_list.append((pixels[w, h][0] + pixels[w, h][1] + pixels[w, h][2]) / int(3))
	pixels_avg_array = np.array(pixels_avg_list)
	pix_mean = pixels_avg_array.mean()
	return  pix_mean

def _color_temp(img):
	img_format = img.format
	width, height = img.size
	pixels = img.load()
	channel_length = len(pixels[0, 0])
	pixels_sum = [0, 0, 0]
	RGB_value = [0, 0, 0]
	for w in range(width):
		for h in range(height):
			if channel_length == 3:
				RGB_value[0], RGB_value[1], RGB_value[2] = pixels[w, h]
			elif channel_length == 4:
				RGB_value = _deal_with_RGBA_image(pixels[w, h])
			for x in range(3):
				pixels_sum[x] += RGB_value[x]
	return pixels_sum

# 處理png圖片
def _deal_with_RGBA_image(RGBA_tuple):
	RGBA_list = [0, 0, 0, 0]
	RGBA_list[0], RGBA_list[1], RGBA_list[2], RGBA_list[3] = RGBA_tuple
	return RGBA_list[:3]

"""
配送方式
會產生2個columns：'12H', 'shopcart'
"""
def transport(soup):
	transportList = [] 
	first = soup.select('#first')
	if first != []:
		transportList.append(1)
	else:
		transportList.append(0)

	shopcart = soup.select('#shopcart')
	if shopcart != []:
		transportList.append(1)
	else:
		transportList.append(0)

	#超商取貨跟超商付款一定同時成立，故只選其一當作feature
	# superstore = soup.select('#superstore')
	# if superstore != []:
	# 	transportList.append(1)
	# else:
	# 	transportList.append(0)

	return transportList

# # 有的尺寸數量
# def productFormatCount(soup):
# 	productFormat = soup.find('select','CompareSel')
# 	productFormatList = productFormat.findAll('option')
# 	productFormatListLen = len(productFormatList)
# 	if productFormatListLen > 1:
# 		productFormatListLen = productFormatListLen-1
# 	return productFormatListLen

# # 在商品規格欄位中有無使用表格
# def attributesListArea(soup):
# 	ListArea = soup.find('div','attributesListArea')
# 	if ListArea != None:
# 		return 1
# 	else:
# 		return 0

# 有無包含影片（一個column）
def haveVideo(soup):
	vendordetailview = soup.find('div', class_='vendordetailview')
	iframe = vendordetailview.find('iframe')
	iframesrc = iframe['src']
	iframe_web = 'https://www.momoshop.com.tw' + iframesrc

	iframe_requests = requests.get(iframe_web, headers=HEADER, cookies=COOKIES)
	iframe_soup = BeautifulSoup(iframe_requests.text, 'html.parser')
	video = iframe_soup.findAll('iframe')

	if len(video) >= 1:
		return 1
	else:
		return 0


# 產地(一個column)
def origin(soup):
	ListArea = soup.find('div','attributesListArea')
	originTypeList = ['產地','原產地','製造','生產','生產地','製造地']

	if ListArea.findAll('th') != []: 
		print("----找表格----")
		ListArea2 = ListArea.findAll('th')
		ListArea3 = ListArea.findAll('ul')
		ListArea2 = [x.text for x in ListArea2]
		ListArea3 = [x.text for x in ListArea3]
		dictionary = dict(list(zip(ListArea2, ListArea3)))
		print(dictionary)
		if '產地' in dictionary:
			print(dictionary['產地'])
			return 1
		elif '商品規格' in dictionary:
		    print(dictionary['商品規格'])
		    dictionary['商品規格'] = dictionary['商品規格'].replace('\n','').replace('　','').replace('\t','').replace(' ','')
		    words = jieba.cut(dictionary['商品規格'], cut_all=False)
		    words = ("/".join(words)).split('/')
		    originTypeIndex = [i for i,v in enumerate(words) if v in originTypeList]
		    if originTypeIndex:
		    	print("originTypeIndex")
		    	return 1
		    else:
		    	return 0
	else:
		return 0

"""
找出銷售單位(瓶or補充包or組合)
產生三個columns：'supplementary', 'bottle', 'combination'
"""
def unit(soup):
	unitList = [0,0,0]
	supplement = ['補充','包','袋']
	bottle = ['瓶','罐']
	ListArea = soup.find('div','attributesListArea')
	# 先找表格中有沒有"單位"欄位
	if ListArea != None:
		ListArea2 = ListArea.findAll('th')
		ListArea3 = ListArea.findAll('ul')
		ListArea2 = [x.text for x in ListArea2]
		ListArea3 = [x.text for x in ListArea3]
		dictionary = dict(list(zip(ListArea2, ListArea3)))
		try:
			unitType = dictionary['單位']
			for i in supplement:
				if i in unitType:
					unitList = [0,1,0]
				elif '組' in unitType:
					unitList = [0,0,1]
				else:
					unitList = [1,0,0]
		except:
			pass
	#若沒找到則比對商品名稱中的關鍵字
	if unitList == [0,0,0]:
		goodName = soup.find('div','prdnoteArea').find('h1').text
		supplementBoolean = False
		bottleBoolean = False
		for i in supplement:
			if i in goodName:
				supplementBoolean = True
		for i in bottle:
			if i in goodName:
				bottleBoolean = True
		if (supplementBoolean and bottleBoolean) or '組' in goodName:
			unitList = [0,0,1]
		elif supplementBoolean:
			unitList = [0,1,0]
		elif bottleBoolean:
			unitList = [1,0,0]
		#若都沒有找到關鍵字，就算是瓶裝
		else:
			unitList = [1,0,0]

	return unitList
