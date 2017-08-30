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

class process_producing_country(object):
    """
    將產地由9維降為2維
    """
    def __init__(self):
        self.count = 0
        self.output_df = pd.DataFrame(columns=['GID', 'price', 'discount', 'payment_CreditCard',
                                'payment_Arrival', 'payment_ConvenienceStore', 'payment_ATM',
                                'payment_iBon', 'preferential_count', 'img_height', 'is_warm',
                                'is_cold', 'is_bright', 'is_dark', '12H', 'shopcart',
                                'superstore', 'productFormatCount', 'attributesListArea',
                                'haveVideo', 'mainProductionPlace', 'otherProductionPlace', 'supplementary', 'bottle',
                                'combination', 'look_times', 'label'])

    def function(self):
        data = pd.read_csv('edition_335/result_detergent_choose335.csv')
        """
        data.shape # (226, 34)
        data.columns[20] # 產地的第一個，台灣
        data.columns[28] # 產地的最後一個，其他
        data.columns[20:29] # 產地全部欄位，共9維
        """
        for index, row in data.iterrows():
            # ===> 有產地，或其他，沒產地就是都0
            found = False
            if row[28] == 1: # 如果是其他
                self.create_df(row[0:20].tolist(), [0, 1], row[29:].tolist(), index)
                continue
            for item in row[20:29]: # 掃過所有產地
                if item != 0: # 有產地
                    found = True
                    self.create_df(row[0:20].tolist(), [1, 0], row[29:].tolist(), index)
                    break
            if found: # 有找到產地
                continue
            else: # 沒有產地
                self.create_df(row[0:20].tolist(), [0, 0], row[29:].tolist(), index)

        self.output_df.to_csv('./edition_226/result_detergent_choose_2dim_226.csv', index=False)

    def create_df(self, head, new_country_list, tail, index):
        # self.count += 1
        row_list = (head + new_country_list + tail)
        self.output_df.loc[index] = row_list

class join_production_country(object):
    def __init__(self):
        pass

    def function(self):
        data = pd.read_csv('./bodywash/result_bodywash_183/result_bodywash_183_withoutPC.csv')
        PC = pd.read_csv('bodywash/production_country/CR_BodyWash_lk10_origin.csv')
        # print PC
        result = pd.merge(data, PC, how='left', on='GID')
        result = result.drop('origin', axis=1)
        result = result[['GID', 'haveOrigin', 'volume', 'supplementary', 'bottle', 'combination',
                        'price', 'discount', 'payment_ConvenienceStore',
                        'preferential_count', 'img_height', 'is_warm', 'is_cold', '12H',
                        'shopcart', 'haveVideo', 'installments', 'look_times', 'label']]
        print result.columns
        result.to_csv('./bodywash/result_bodywash_183/result_bodywash_183.csv', index=False)



if __name__ == '__main__':
    # obj = process_description()
    # obj.get_seg_list('溫柔洗淨衣物，呵護你的肌膚?')
    # obj.high_and_low_kw()
    obj = join_production_country()
    obj.function()


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

