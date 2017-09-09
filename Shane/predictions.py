# -*- coding: utf-8 -*-
import pandas as pd
from predict import *

##############################
##         洗衣精區          ##
##############################

# --------------- 初始898版本 (898筆，爬的到685多筆，濾完looktime剩119筆) ---------------
# data = pd.read_csv('./detergent_new_price/result_detergent_898_4.csv')
# label = pd.read_csv('winning_index/ProValue_4.0.csv')
# data = join_label(data, label)


# --------------- 有加入「有買但沒看過的」，濾完looktime有335筆，爬的到有226筆 ---------------
# new_price_and_label = pd.read_csv('./detergent_new_price/proValue_detergent_para1_month3.csv')
# new_price_and_label = pd.read_csv('./detergent_new_price/choose_rate_detergent_month3.csv')
# data = pd.read_csv('./0818/result_detergent_choose0817.csv') # 已經濾過look_time
# new_price_and_label = pd.read_csv('./0818/winning_index_3month_0818.csv')
# new_price_and_label = pd.read_csv('./0818/choose_rate_3month_0818.csv')


# --------------- 310版本 (濾完looktime有310筆，爬的到的有198筆) ---------------
# data = pd.read_csv('./edition_310/result_detergent_310.csv')

# data = pd.read_csv('./edition_226/result_detergent_choose_2dim_226.csv')
# new_price_and_label = pd.read_csv('./edition_226/winning_index_3month_looktimes60_0824.csv')


##############################
##         沐浴乳區          ##
##############################

# --------------- 沐浴乳，觀看次數10，獲選率(183筆)---------------
# data = pd.read_csv('./bodywash/CR/result_bodywash_183.csv')
# new_price = pd.read_csv('./bodywash/WI/WI_bodywash_3m_lk10.csv')
# data = join_price_for_CR(data, new_price)
# data = drop_something(data, ['GID', 'look_times'])

# --------------- 沐浴乳，觀看次數20，獲選率(95筆)---------------
# data = pd.read_csv('./bodywash/CR/result_bodywash_183.csv')
# new_price = pd.read_csv('./bodywash/WI/WI_bodywash_3m_lk20.csv')
# data = join_price_for_CR(data, new_price)
# data = drop_something(data, ['GID', 'look_times'])

# --------------- 沐浴乳，觀看次數30，獲選率(61筆)---------------
data = pd.read_csv('./bodywash/CR/result_bodywash_183.csv')
new_price = pd.read_csv('./bodywash/WI/WI_bodywash_3m_lk30.csv')
data = join_price_for_CR(data, new_price)
data = drop_something(data, ['GID', 'look_times'])


# --------------- 沐浴乳，勝敗指數觀看次數10，有join來自goods_describe的price (183筆)---------------
# data = pd.read_csv('./bodywash/CR/result_bodywash_183.csv')
# new_price_and_label = pd.read_csv('./bodywash/WI/WI_bodywash_3m_lk10.csv')
# data = join_new_label_and_price(data, new_price_and_label)
# data = drop_something(data, ['GID', 'look_times'])


# --------------- 沐浴乳，勝敗指數觀看次數20，有join來自goods_describe的price (95筆)---------------
# data = pd.read_csv('./bodywash/CR/result_bodywash_183.csv')
# new_price_and_label = pd.read_csv('./bodywash/WI/WI_bodywash_3m_lk20.csv')
# data = join_new_label_and_price(data, new_price_and_label)
# data = drop_something(data, ['GID', 'look_times'])

# --------------- 沐浴乳，勝敗指數觀看次數30，有join來自goods_describe的price (61筆)---------------
# data = pd.read_csv('./bodywash/CR/result_bodywash_183.csv') 
# new_price_and_label = pd.read_csv('./bodywash/WI/WI_bodywash_3m_lk30.csv')
# data = join_new_label_and_price(data, new_price_and_label)
# data = drop_something(data, ['GID', 'look_times'])


# data = standardizing(data)
data = normalizing(data)
print 'Shape: ', data.shape

# scatter_plots(data)
# activate_drop_columns(data)

ridge_regression(data, 0.1)
# SVR(data)
