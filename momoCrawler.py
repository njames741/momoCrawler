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

def timeit(method):
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()

		print '執行 %r 函式，共花 %2.2f sec' % \
			  (method.__name__, te-ts)
		return result

	return timed

class momo(object):
	def __init__(self):
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
		self.result_df = pd.DataFrame(columns=('GID', 'price', 'discount', 'payment_CreditCard', 'payment_Arrival', 'payment_ConvenienceStore', 'payment_ATM', 'payment_iBon', 'preferential_count', 'reciprocal', 'img_height', 'is_warm', 'is_cold', 'is_bright', 'is_dark', 'label'))

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

		time.sleep(3)
		iframe_requests = requests.get(iframe_web, headers=self.headers, cookies=self.cookies)
		iframe_soup = BeautifulSoup(iframe_requests.text, 'html.parser')

		opener = urllib2.build_opener()
		opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
		imgs = iframe_soup.find_all('img')

		if len(imgs) == 0:
			return 'No Image', 'No Image', 'No Image'

		height_sum = 0
		r_sum, b_sum = 0, 0
		brightness_sum = 0

		for img in imgs:
			imgsrc = img['src'].split('?')[0]
			if imgsrc[:6] != 'https:':
				imgsrc = 'https://www.momoshop.com.tw' + imgsrc

			if imgsrc[8:12] == 'img1' or imgsrc[8:12] == 'img2':
				imgsrc = 'https://img3' + imgsrc[12:]

			try:
				image_file = opener.open(imgsrc)
				print imgsrc
			except:
				print '==============img url錯誤===================='
				print imgsrc
			temp_image = cStringIO.StringIO(image_file.read())
			image = Image.open(temp_image)

			# 處理色溫
			r_sum += self.color_temp(image)[0]
			b_sum += self.color_temp(image)[2]

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
		image_pixels = list()
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
		image_pixels = list()
		width, height = img.size
		pixels = img.load()
		pixels_sum = [0, 0, 0]
		RGB_value = [0, 0, 0]
		for w in range(width):
			for h in range(height):
				RGB_value[0], RGB_value[1], RGB_value[2] = pixels[w, h]
				for x in range(3):
					pixels_sum[x] += RGB_value[x]
		return pixels_sum
	
	@timeit
	def get_rows(self, goods_icode, label):
		web = 'https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code=' + goods_icode
		time.sleep(3)
		h = requests.get(web, headers=self.headers, cookies=self.cookies)
		soup = BeautifulSoup(h.text, 'html.parser')

		row_list = list()
		row_list.append(goods_icode)
		row_list.append(self.price(soup))
		row_list.append(self.discount(soup))
		row_list += self.payment(soup)
		row_list.append(self.preferentialCount(soup))
		row_list.append(self.reciprocal(soup))
		img_result_list = self.image_analysis(soup)
		row_list.append(img_result_list[0])
		row_list += img_result_list[1]
		row_list += img_result_list[2]
		row_list.append(label)
		return row_list

	def create_csv(self):
		gid_list = pd.read_csv('./data_diamond_with_label.csv').values
		requests_count = 0
		row_index = 0
		abandoned = 0
		for row in gid_list:
			print '---------------------------'
			requests_count += 1
			if requests_count == 2: break
			print str(int(row[0])), row[1]
			try:
				self.result_df.loc[row_index] = self.get_rows(str(int(row[0])), row[1])
				row_index += 1
				print '已requests數: ', requests_count
			except:
				abandoned += 1
				print '爬不到，抱歉'
				print '已requests數: ', requests_count
				continue

		print '爬不到的頁面總數量: ', abandoned
		self.result_df.to_csv('./result_jewelry_3264.csv', index=False)

	def testing(self):
		# gid_label_list = pd.read_csv('data_diamond_with_label.csv').values
		# pprint(gid_label_list)
		# for row in gid_label_list:
			# print str(int(row[0])), row[1]
			# print self.get_rows(str(int(row[0])), row[1])
		# string = '3189401'
		# print self.get_rows(string, 0.344444)
			# break
		print self.result_df.columns


if __name__ == '__main__':
	import time
	import sys
	obj = momo()

	start = time.time()
	obj.create_csv()
	end = time.time()

	time_cost = end - start
	print "總花費時間", time_cost, "秒"
	# obj.get_rows(sys.argv[1])
	# obj = momo()
	# obj.testing()
