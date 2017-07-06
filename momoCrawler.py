# -*- coding: utf-8 -*-

import requests
import _uniout
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
import pytesseract
import time

class momoGet(object):

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

	def price(self,soup):
		try:
			price = soup.find('li','special').find('span').text
		except:
			price = soup.find('ul' ,'prdPrice').find('li').find('del').text
		price = price.replace(",","")
		print "price: ",int(price)

	def discount(self,soup):
		try:
			OldPrice = soup.find('ul' ,'prdPrice').find('li').find('del').text.replace(",","")
			NewPrice = soup.find('li','special').find('span').text.replace(",","")
			print "discount: ",int(OldPrice) - int(NewPrice)
		except:
			print "discount: ",0

	def payment(self,soup):
		paymentList = [u'信用卡',u'貨到付款', u'超商付款', u'ATM', u'iBon']
		paymentFeature = list()
		payment = soup.find('dl','payment').text.split("\n")
		for i in paymentList:
			if i in payment:
				paymentFeature.append(1)
			else:
				paymentFeature.append(0)

		print "payment",paymentFeature
		# print payment[1].text
		# print payment[2].text

	def preferentialCount(self,soup):
		preferential = soup.find('dl','preferential').findAll('dd')
		# for i in preferential:
		#   print i.text
		print "preferentialCount: ", len(preferential)


	def image_analysis(self, soup):
		# vendordetailview 是整個「商品特色」頁面的標籤
		vendordetailview = soup.find('div', class_='vendordetailview')
		iframe = vendordetailview.find('iframe')
		iframesrc = iframe['src']
		print "iframesrc: ",iframesrc
		iframe_web = 'https://www.momoshop.com.tw' + iframesrc
		print "iframe_web: ",iframe_web
		# time.sleep(5)
		iframe_requests = requests.get(iframe_web, headers=self.headers2, cookies=self.cookies2)
		iframe_soup = BeautifulSoup(iframe_requests.text, 'html.parser')

		opener = urllib2.build_opener()
		opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
		imgs = iframe_soup.find_all('img')
		height_sum = 0
		for img in imgs:
			# time.sleep(5)
			imgsrc = img['src'].split('.jpg')[0] + '.jpg'
			if imgsrc[:6] != 'https:':
				imgsrc = 'https:' + imgsrc
			print imgsrc
			image_file = opener.open(imgsrc)
			temp_image = cStringIO.StringIO(image_file.read())
			image = Image.open(temp_image)
			# image.show()
			width, height = image.size
			height_sum += height
		print height_sum

	def main(self,goods_icode):
		web = 'https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code=' + goods_icode
		h = requests.get(web, headers=self.headers, cookies=self.cookies)
		soup = BeautifulSoup(h.text, 'lxml')
		self.price(soup)
		self.discount(soup)
		self.payment(soup)
		self.preferentialCount(soup)
		self.image_analysis(soup)


		# image = Image.open('/home/nj/workspace/momo/image1.jpg')
		# number = pytesseract.image_to_string(image,lang ='chi_tra')
		# print "The captcha is:%s" % number

if __name__ == '__main__':
	obj = momoGet()
	obj.main("4856789")