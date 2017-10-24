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

# --------------- 310版本 (濾完looktime有310筆，爬的到的有198筆) ---------------
# data = pd.read_csv('./edition_310/result_detergent_310.csv')

# data = pd.read_csv('./edition_226/result_detergent_choose_2dim_226.csv')
# new_price_and_label = pd.read_csv('./edition_226/winning_index_3month_looktimes60_0824.csv')

# -------------------------------------- CM ---------------------------------------
# data = pd.read_csv('./detergent/edition_226/result_detergent_CR_226.csv') # 已經濾過look_time
# new_price_and_label = pd.read_csv('./detergent/edition_226/winning_index_3month_looktimes60_0824.csv')
# data = join_new_label_and_price(data, new_price_and_label)
# data = drop_something(data, ['GID', 'look_times'])


##############################
##         沐浴乳區          ##
##############################

# --------------- 沐浴乳，觀看次數10，獲選率(183筆)---------------
# data = pd.read_csv('./bodywash/CR/result_bodywash_183.csv')
# new_price = pd.read_csv('./bodywash/WI/WI_bodywash_3m_lk30.csv')
# data = join_price_for_CR(data, new_price)
# data = drop_something(data, ['GID', 'look_times'])


# --------------- 沐浴乳，勝敗指數---------------
data = pd.read_csv('./bodywash/CR/result_bodywash_183.csv')
new_price_and_label = pd.read_csv('./bodywash/WI/WI_bodywash_3m_lk50.csv')
data = join_new_label_and_price(data, new_price_and_label)
data = drop_something(data, ['GID', 'look_times'])


##############################
##         精華液區          ##
##############################
# data = pd.read_csv('./essense/result_essence_WI_lk30.csv')
# new_price_and_label = pd.read_csv('./essense/WI_essence_3m_lk40.csv')
# data = join_new_label_and_price(data, new_price_and_label)
# data = drop_something(data, ['GID', 'look_times'])



##############################
##        洗衣精新嘗試        ##
##############################
# data = pd.read_csv('./detergent/edition_226/result_detergent_CR_226.csv')

# new_price_and_label = pd.read_csv('./new_try/WI_detergent_3m_lk20_pr.csv')
# new_price_and_label = pd.read_csv('./new_try/WI_detergent_3m_lk60_pr_rt.csv')
# new_price_and_label = pd.read_csv('./edition_226/winning_index_3month_looktimes60_0824.csv') # 勝敗指數lt60

# new_price_and_label = pd.read_csv('./detergent/edition_226/WI_detergent_3m_lk60_v2.csv') # v2
# data = join_new_label_and_price(data, new_price_and_label)
# data = drop_something(data, ['GID', 'look_times'])




# 算corr
# pr = pd.read_csv('./new_try/WI_detergent_3m_lk20_pr.csv')
# cm = pd.read_csv('./edition_226/winning_index_3month_looktimes20_0824.csv')
# pr = pd.read_csv('./new_try/WI_detergent_3m_lk60_pr_rt.csv')
# cr = pd.read_csv('./edition_226/result_detergent_choose_226.csv')
# print cr[['GID', 'label']]
# result = pd.merge(pr, cr[['GID', 'label']], on='GID')
# print result
# print result[['label_x', 'label_y']].corr()


# data = standardizing(data)
# data = normalizing(data)
data = normalizing_with_label(data)
# data.to_csv('./temp.csv')
print 'Shape: ', data.shape
# print data.columns

# scatter_plots(data)
# activate_drop_columns(data)

ridge_regression(data, 0.1)
# SVR(data)
