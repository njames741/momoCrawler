# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
from pprint import pprint
import re
import sys
import cStringIO
import urllib2
from PIL import Image
import pandas as pd
import csv
import os.path
import time
import numpy as np
import jieba
import traceback



def timeit(method):
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()
		print '執行 %r 函式，共花 %2.2f sec' % (method.__name__, te-ts)
		return result
	return timed

class SystemInputError(Exception):
    pass

class momo(object):
	def __init__(self, status):
		self.headers = {
			'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1700.107 Safari/537.36',
		}
		self.headers2 = {
			'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36',
		}
		self.cookies = {
			'_ts_id': '999999999999999999',
		}
		self.cookies2 = {
			'_ts_id': '888888888888888888',
		}
		self.result_df = pd.DataFrame(columns=('GID', 'price', 'discount', 'payment_CreditCard', \
			'payment_Arrival', 'payment_ConvenienceStore', 'payment_ATM', 'payment_iBon', \
			'preferential_count', 'img_height', 'is_warm', 'is_cold', 'is_bright', 'is_dark', \
			'12H', 'shopcart', 'superstore', 'productFormatCount', 'attributesListArea', \
			'haveVideo', 'Taiwan','EUandUS','Germany','UK','US','Japan','Malaysia','Australia','other', \
		 	'supplementary', 'bottle', 'combination', 'look_times', 'label'))
		# outputOriginList = [u'台灣', u'歐美', u'德國', u'英國', u'美國', u'日本', u'馬來西亞', u'澳洲', u'其他']
		if status == 'c':
			self.with_header = False
		elif status == 'i':
			self.with_header = True
		else:
			raise SystemInputError('系統參數請輸入: c -> 續寫, i -> 從頭開始執行')

		jieba.set_dictionary('dict.txt.big')

	# 價錢
	def price(self,soup):
		try:
			price = soup.find('li','special').find('span').text
		except:
			price = soup.find('ul' ,'prdPrice').find('li').find('del').text
		price = price.replace(",","")
		# print "price: ",int(price)
		return int(price)

	# 折扣
	def discount(self,soup):
		try:
			OldPrice = soup.find('ul' ,'prdPrice').find('li').find('del').text.replace(",","")
			NewPrice = soup.find('li','special').find('span').text.replace(",","")
			# print "discount: ",int(OldPrice) - int(NewPrice)
			return (int(OldPrice) - int(NewPrice))
		except:
			# print "discount: ",0
			return 0

	# 付款方式(one hot)
	def payment(self,soup):
		paymentList = [u'信用卡',u'貨到付款', u'超商付款', u'ATM', u'iBon']
		paymentFeature = list()
		payment = soup.find('dl','payment').text.split("\n")
		for i in paymentList:
			if i in payment:
				paymentFeature.append(1)
			else:
				paymentFeature.append(0)
		# print "payment",paymentFeature
		return paymentFeature

	# 贈品(數量)
	def preferentialCount(self,soup):
		try:
			preferential = soup.find('dl','preferential').findAll('dd')
			return len(preferential)
		except:
			return 0

	# 庫存倒數
	def reciprocal(self,soup):
		reciprocal = soup.select('#goodsDtCount_001')[0]['value']
		if int(reciprocal) <= 5:
			return 1
		else:
			return 0

	@timeit
	def image_analysis(self, soup):
		# vendordetailview 是整個「商品特色」頁面的標籤
		vendordetailview = soup.find('div', class_='vendordetailview')
		iframe = vendordetailview.find('iframe')
		iframesrc = iframe['src']
		iframe_web = 'https://www.momoshop.com.tw' + iframesrc

		# time.sleep(3)
		iframe_requests = requests.get(iframe_web, headers=self.headers, cookies=self.cookies)
		iframe_soup = BeautifulSoup(iframe_requests.text, 'html.parser')

		opener = urllib2.build_opener()
		opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
		imgs = iframe_soup.find_all('img')

		if len(imgs) == 0:
			return 0, [0, 0], [0, 0]

		height_sum = 0
		r_sum, b_sum = 0, 0
		brightness_sum = 0

		for img in imgs:
			print img['src']
			imgsrc = img['src'].split('?')[0]
			if imgsrc[:6] == '/exper':
				imgsrc = 'https://www.momoshop.com.tw' + imgsrc
			if imgsrc[:5] == '//img':
				imgsrc = 'https:' + imgsrc

			if imgsrc[8:12] == 'img1' or imgsrc[8:12] == 'img2':
				imgsrc = 'https://img3' + imgsrc[12:]
			imgsrc = imgsrc.replace('"','')
			# req = urllib2.Request(imgsrc,headers=self.headers)
			try:
				image_file = opener.open(imgsrc)
				print imgsrc
			except Exception as e:
				print '==============img url錯誤===================='
				traceback.print_exc()
				print imgsrc
			temp_image = cStringIO.StringIO(image_file.read())
			image = Image.open(temp_image)

			# 處理色溫
			rgb_sum_list = self.color_temp(image)
			r_sum += rgb_sum_list[0]
			b_sum += rgb_sum_list[2]

			# 處理高度
			width, height = image.size
			height_sum += height

			# 處理亮度
			brightness_sum += self.get_brightness(image)

		# 色溫
		if r_sum > b_sum: temp_list = [1, 0]
		else: temp_list = [0, 1]

		# 平均亮度計算
		brightness_avg = brightness_sum / len(imgs)
		if brightness_avg > 127: brightness_list = [1, 0]
		else: brightness_list = [0, 1]

		# print '圖片高度: ', height_sum
		# print '色溫', temperature # 暖是1，暗是0
		# print '亮度', brightness # 亮是1，暗是0
		return height_sum, temp_list, brightness_list

	def get_brightness(self, img):
		width, height = img.size
		pixels = img.load()
		pixels_avg_list = list()
		for w in range(width):
			for h in range(height):
				pixels_avg_list.append((pixels[w, h][0] + pixels[w, h][1] + pixels[w, h][2]) / int(3))
		pixels_avg_array = np.array(pixels_avg_list)
		pix_mean = pixels_avg_array.mean()
		return  pix_mean

	def color_temp(self, img):
		img_format = img.format
		# print img_format
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
					RGB_value = self.deal_with_RGBA_image(pixels[w, h])
				for x in range(3):
					pixels_sum[x] += RGB_value[x]
		return pixels_sum

	# 處理png圖片
	def deal_with_RGBA_image(self, RGBA_tuple):
		RGBA_list = [0, 0, 0, 0]
		RGBA_list[0], RGBA_list[1], RGBA_list[2], RGBA_list[3] = RGBA_tuple
		return RGBA_list[:3]

	# 配送方式
	def transport(self,soup):
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

		superstore = soup.select('#superstore')
		if superstore != []:
			transportList.append(1)
		else:
			transportList.append(0)

		return transportList

	# 有的尺寸數量
	def productFormatCount(self, soup):
		productFormat = soup.find('select','CompareSel')
		productFormatList = productFormat.findAll('option')
		productFormatListLen = len(productFormatList)
		if productFormatListLen > 1:
			productFormatListLen = productFormatListLen-1
		return productFormatListLen

	#在商品規格欄位中有無使用表格
	def attributesListArea(self, soup):
		ListArea = soup.find('div','attributesListArea')
		if ListArea != None:
			return 1
		else:
			return 0

	#有無包含影片
	def haveVideo(self, soup):
		vendordetailview = soup.find('div', class_='vendordetailview')
		iframe = vendordetailview.find('iframe')
		iframesrc = iframe['src']
		iframe_web = 'https://www.momoshop.com.tw' + iframesrc

		# time.sleep(3)
		iframe_requests = requests.get(iframe_web, headers=self.headers, cookies=self.cookies)
		iframe_soup = BeautifulSoup(iframe_requests.text, 'html.parser')
		video = iframe_soup.findAll('iframe')

		if len(video) >= 1:
			return 1
		else:
			return 0


	#產地
	def origin(self, soup):
		ListArea = soup.find('div','attributesListArea')
		specificationArea = soup.find('div','vendordetailview specification')
		specificationArea = specificationArea.find('p')

		originList = [u'台灣',u'臺灣',u'德國',u'英國',u'歐美',u'歐洲',u'日本',u'美國',u'其他',u'其它',u'馬來西亞'\
					,u'法國',u'東南亞',u'亞州',u'韓國',u'中國',u'大陸',u'中國大陸',u'澳洲']
		originTypeList = [u'產地',u'原產地',u'製造',u'生產',u'生產地',u'製造地']

		outputOriginList = [u'台灣', u'歐美', u'德國', u'英國', u'美國', u'日本', u'馬來西亞', u'澳洲', u'其他']
		outputList = [0,0,0,0,0,0,0,0,0]
		finalOrigin = ''

		vendordetailview = soup.find('div', class_='vendordetailview')
		iframe = vendordetailview.find('iframe')
		iframesrc = iframe['src']
		iframe_web = 'https://www.momoshop.com.tw' + iframesrc
		iframe_requests = requests.get(iframe_web, headers=self.headers, cookies=self.cookies)
		iframe_soup = BeautifulSoup(iframe_requests.text, 'html.parser')
		iframe_soup = iframe_soup.text.replace('\n','').replace(' ','')
		iframeWords = jieba.cut(iframe_soup, cut_all=False)
		iframeWords = ("/".join(iframeWords)).split('/')
		originTypeIndexiframe = [i for i,v in enumerate(iframeWords) if v in originTypeList]

		specificationArea = specificationArea.text.replace('\n','').replace(' ','')
		words = jieba.cut(specificationArea, cut_all=False)
		words = ("/".join(words)).split('/')
		#找各種產地字詞的index
		originTypeIndex = [i for i,v in enumerate(words) if v in originTypeList]
		# for i in iframeWords:
		# 	print i
		
		# print originTypeIndexiframe

		#先找表格下面的文字中有無產地
		if originTypeIndex:
			temp = []
			for i in originTypeIndex:
				if (i-6) < 0:
					start = 0
				else:
					start = (i-6)
				temp += words[start:i+6]
			temp = list(set(temp))
			# for i in temp:
			# 	print i
			origin =  [val for val in originList if val in temp]
			if origin:
				finalOrigin = origin[0]
		#再找表格
		print ListArea.findAll('th')
		if ListArea.findAll('th') != [] and finalOrigin == u'': 
			ListArea2 = ListArea.findAll('th')
			ListArea3 = ListArea.findAll('ul')
			ListArea2 = map(lambda x:x.text,ListArea2)
			ListArea3 = map(lambda x:x.text,ListArea3)
			dictionary = dict(zip(ListArea2, ListArea3))
			print dictionary
			finalOrigin = dictionary[u'產地']
		#再找商品特色頁面
		if originTypeIndexiframe and finalOrigin == u'':
			temp = []
			for i in originTypeIndexiframe:
				if (i-6) < 0:
					start = 0
				else:
					start = (i-6)
				temp += iframeWords[start:i+6]
			temp = list(set(temp))
			origin =  [val for val in originList if val in temp]
			if origin:
				finalOrigin = origin[0]	
		if finalOrigin == u'':
			finalOrigin = u'N'
			
		for i in range(len(outputOriginList)):
			if finalOrigin == outputOriginList[i]:
				outputList[i] = 1

		return outputList

	#找出銷售單位(瓶or補充包or組合)
	def unit(self, soup):
		unitList = [0,0,0]
		supplement = [u'補充',u'包',u'袋']
		bottle = [u'瓶',u'罐']
		ListArea = soup.find('div','attributesListArea')
		#先找表格中有沒有"單位"欄位
		if ListArea != None:
			ListArea2 = ListArea.findAll('th')
			ListArea3 = ListArea.findAll('ul')
			ListArea2 = map(lambda x:x.text,ListArea2)
			ListArea3 = map(lambda x:x.text,ListArea3)
			dictionary = dict(zip(ListArea2, ListArea3))
			try:
				unitType = dictionary[u'單位']
				for i in supplement:
					if i in unitType:
						unitList = [0,1,0]
					elif u'組' in unitType:
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
			if (supplementBoolean and bottleBoolean) or u'組' in goodName:
				unitList = [0,0,1]
			elif supplementBoolean:
				unitList = [0,1,0]
			elif bottleBoolean:
				unitList = [1,0,0]
			#若都沒有找到關鍵字，就算是瓶裝
			else:
				unitList = [1,0,0]

		return unitList

	@timeit
	def get_rows(self, goods_icode, look_num, label):
		web = 'https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code=' + goods_icode
		time.sleep(1)
		h = requests.get(web, headers=self.headers, cookies=self.cookies)
		soup = BeautifulSoup(h.text, 'html.parser')

		row_list = list()
		row_list.append(goods_icode)
		row_list.append(self.price(soup))
		row_list.append(self.discount(soup))
		row_list += self.payment(soup)
		row_list.append(self.preferentialCount(soup))
		img_result_list = self.image_analysis(soup)
		row_list.append(img_result_list[0])
		row_list += img_result_list[1]
		row_list += img_result_list[2]
		row_list += self.transport(soup)
		row_list.append(self.productFormatCount(soup))
		row_list.append(self.attributesListArea(soup))
		row_list.append(self.haveVideo(soup))
		row_list += self.origin(soup)
		row_list += self.unit(soup)
		row_list.append(look_num)
		row_list.append(label)
		


		print row_list

		# for key,value in row_list[0].items():
		# 	print key,value 

		# return row_list

	def create_csv(self, input_file_name, output_file_name):
		gid_list = pd.read_csv(input_file_name).values
		requests_count = 0
		successful = 0
		abandoned = 0
		no_page_count = 0
		first_write = True
		for row in gid_list:
			print '---------------------------'
			requests_count += 1
			# if requests_count == 10: break
			print str(int(row[0])), row[1]
			try:
				self.result_df.loc[0] = self.get_rows(str(int(row[0])), row[3], row[1])
				if first_write and self.with_header:
					self.result_df.to_csv(output_file_name, mode='a', index=False)
					first_write = False
					self.with_header = False
				else:
					self.result_df.to_csv(output_file_name, mode='a', index=False, header=False)
				successful += 1
				print '已requests數: ', requests_count
				print '已有資料筆數: ', successful

			except Exception as e:
				abandoned += 1
				print '爬不到，抱歉'
				# print e
				traceback.print_exc()
				if str(e) == "'NoneType' object has no attribute 'find'":
					no_page_count += 1
				print '已requests數: ', requests_count
				print '已有資料筆數: ', successful
				continue

		print '處理失敗總數量: ', abandoned
		print '無頁面總數量: ', no_page_count

	# def testing(self, img_url):
	# 	opener = urllib2.build_opener()
	# 	opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
	# 	image_file = opener.open(img_url)
	# 	temp_image = cStringIO.StringIO(image_file.read())
	# 	img = Image.open(temp_image)
	# 	print self.color_temp_2(img)

	# def color_temp_2(self, img):
	# 	img_format = img.format
	# 	print img_format
	# 	width, height = img.size
	# 	pixels = img.load()
	# 	pixels_sum = [0, 0, 0]
	# 	RGB_value = [0, 0, 0]
	# 	print pixels[45, 23]
	# 	for w in range(width):
	# 		for h in range(height):
	# 			if img_format == 'JPEG':
	# 				RGB_value[0], RGB_value[1], RGB_value[2] = pixels[w, h]
	# 			elif img_format == 'PNG':
	# 				RGB_value = self.deal_with_RGBA_image(pixels[w, h])
	# 			else:
	# 				print '影像格式不是JPEG或PNG'
	# 				RGB_value[0], RGB_value[1], RGB_value[2] = pixels[w, h]
	# 			for x in range(3):
	# 				pixels_sum[x] += RGB_value[x]
	# 	return pixels_sum

if __name__ == '__main__':
	import time
	import sys
	# 如果是從某個GID開始續寫，輸入小寫c
	# 如果是從頭開始跑，輸入小寫i
	# obj = momo(sys.argv[1])

	# start = time.time()
	# obj.create_csv(sys.argv[2], sys.argv[3])
	# end = time.time()

	# time_cost = end - start
	# print "總花費時間", time_cost, "秒"

	obj = momo('i')
	obj.get_rows(sys.argv[1],123,321)
