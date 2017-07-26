# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
import pandas as pd
from pprint import pprint

class process_description(object):
    def __init__(self, data_path='new_detergent_goods_des.csv'):
        self.data = pd.read_csv(data_path)

    def function(self):
        pages_text_list = []

        pages_count = self.data.shape[0]
        pages_with_kw_count = 0
        process_count = 0
        for item in self.data[['DESCRIBE_301', 'DESCRIBE_302']].values.tolist():
            make_it_single_string = '' # 一個商品頁面的description
            process_count += 1
            make_it_single_string += (str(item[0]) + str(item[1]))
            pages_with_kw_count += self.if_contains(make_it_single_string)
            if process_count%10 == 0: print '已處理', process_count, '筆'
            # break
        print pages_with_kw_count, '/', pages_count


        # print make_it_single_string
        # self.get_keywords(make_it_single_string)

    def get_keywords(self, all_text):
        kw_list = jieba.analyse.extract_tags(all_text, topK=10, withWeight=False, allowPOS=())
        # return set(kw_list)
        for kw in kw_list:
            print kw


    def if_contains(self, one_page_des):
        kw_dict_high_ratio = {u'配方': 0, u'天然': 0, u'洗淨': 0, u'衣物': 0, u'認證': 0, u'酵素': 0, u'抗菌': 0, u'螢光劑': 0, u'進口': 0, u'升級': 0}
        kw_dict_low_ratio = {u'配方': 0, u'天然': 0, u'洗淨': 0, u'衣物': 0, u'認證': 0, u'酵素': 0, u'抗菌': 0, u'螢光劑': 0, u'進口': 0, u'升級': 0}
        # kw_dict = {u'酵素'}
        # kw_dict = {u'唬爛啦'}
        seg_list = jieba.lcut(one_page_des, cut_all=False)
        for item in seg_list:
            if item in kw_dict:
                # print '壞了'
                return 1
        # print '沒壞，真的沒這東西'
        return 0

    def high_and_low_kw(self):
        des_data = self.data[['rate', 'DESCRIBE_301', 'DESCRIBE_302']]
        high_ratio_df = des_data.loc[des_data['rate'] >= 0.111, :]
        low_ratio_df = des_data.loc[des_data['rate'] < 0.111, :]

        high_ratio_text = self.get_all_text_str(high_ratio_df)
        low_ratio_text = self.get_all_text_str(low_ratio_df)

        print '-----較高獲選率-----'
        high_set = self.get_keywords(high_ratio_text)
        print '-----較低獲選率-----'
        low_set = self.get_keywords(low_ratio_text)
        

    def get_all_text_str(self, df):
        all_text = '' # 所有商品的文字加起來
        for item in df[['DESCRIBE_301', 'DESCRIBE_302']].values.tolist():
            all_text += (str(item[0]) + str(item[1]))
        return all_text
            
        



        

if __name__ == '__main__':
    obj = process_description()
    # obj.get_seg_list('溫柔洗淨衣物，呵護你的肌膚?')
    obj.high_and_low_kw()


"""
new_detergent_goods_des.csv 有890筆

:::獲選率統計資料:::
             rate
count  890.000000
mean     0.193285
std      0.242556
min      0.000000
25%      0.000000
50%      0.111111
75%      0.333333
max      1.000000


"""

