"""
momo類別進行以下工作
- momo購物網爬蟲
- 由 feature_extraction 模塊進行特徵萃取工作
- 產生特徵表

直接執行此檔案的使用方式：
需輸入三個系統參數
- arg1 : 字母 i 或 c，如果是第一次產生資料表，輸入 i ，若是爬蟲到一半斷掉，可輸入 c 續寫資料表
- arg2 : 輸入檔名（輸入檔格式參考create_csv函式的註解）
- arg3 : 輸出檔名
"""
import requests
from bs4 import BeautifulSoup
from pprint import pprint
import sys
import pandas as pd
import csv
import os.path
import time
import traceback

from feature_extraction import *
from request_config import HEADER, COOKIES

# 計算執行時間
def timeit(method):
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()
		print(('跑完 %r 函式 , 共花 %2.2f sec' % (method.__name__, te-ts)))
		return result
	return timed

class SystemInputError(Exception):
    pass

class momo(object):
	def __init__(self, status):
		self.result_df = pd.DataFrame(columns=('GID', 'price', 'discount', 'payment_CreditCard', \
			'payment_Arrival', 'payment_ConvenienceStore', 'payment_ATM', 'payment_iBon', \
			'preferential_count', 'img_height', 'is_warm', 'is_cold', 'is_bright', 'is_dark', \
			'12H', 'shopcart', 'superstore', 'productFormatCount', 'attributesListArea', \
			'haveVideo', 'Taiwan','EUandUS','Germany','UK','US','Japan','Malaysia','Australia','other', \
		 	'supplementary', 'bottle', 'combination', 'look_times', 'label'))
		if status == 'c':
			self.with_header = False
		elif status == 'i':
			self.with_header = True
		else:
			raise SystemInputError('系統參數請輸入: c -> 續寫, i -> 從頭開始執行')

	@timeit
	def get_rows(self, goods_icode, look_num, label):
		web = 'https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code=' + goods_icode
		time.sleep(1)
		h = requests.get(web, headers=HEADER, cookies=COOKIES)
		soup = BeautifulSoup(h.text, 'html.parser')

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
		print(row_list)

		# return row_list

	def create_csv(self, input_file_name, output_file_name):
		"""
		Args:
			- 輸入檔名 (輸入為csv檔，三個columns分別為  1. GID 2. 計算完成的Competitiveness Metric 3. 來自op21資料表的真實price值)
			- 輸出檔名
		Output:
			- 完整特徵與Label資料表，可直接用於模型訓練
		"""
		gid_list = pd.read_csv(input_file_name).values
		requests_count = 0
		successful = 0
		abandoned = 0
		no_page_count = 0
		first_write = True
		for row in gid_list:
			print('---------------------------')
			requests_count += 1
			print((str(int(row[0])), row[1]))
			try:
				self.result_df.loc[0] = self.get_rows(str(int(row[0])), row[3], row[1])
				if first_write and self.with_header:
					self.result_df.to_csv(output_file_name, mode='a', index=False)
					first_write = False
					self.with_header = False
				else:
					self.result_df.to_csv(output_file_name, mode='a', index=False, header=False)
				successful += 1
				print(('已requests數: ', requests_count))
				print(('已有資料筆數: ', successful))

			except Exception as e:
				abandoned += 1
				print('爬不到，抱歉')
				traceback.print_exc()
				if str(e) == "'NoneType' object has no attribute 'find'":
					no_page_count += 1
				print(('已requests數: ', requests_count))
				print(('已有資料筆數: ', successful))
				continue

		print(('處理失敗總數量: ', abandoned))
		print(('無頁面總數量: ', no_page_count))


if __name__ == '__main__':
	import time
	import sys
	
	# obj = momo(sys.argv[1])

	# start = time.time()
	# obj.create_csv(sys.argv[2], sys.argv[3])
	# end = time.time()

	# time_cost = end - start
	# print "總花費時間", time_cost, "秒"

	obj = momo('i')
	obj.get_rows('3812355', 123, 321)
