import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split

# items_categorical = pd.read_csv('item_categories.csv')
# items = pd.read_csv('items.csv')
# shops = pd.read_csv('shops.csv')


train = pd.read_csv('sales_train.csv')
test = pd.read_csv('test.csv')

train = train.loc[train.item_cnt_day > 0.]
train = train.loc[train.item_cnt_day < 5.0]
train = train.loc[train.item_price < 500.]
train = train.loc[train.item_price > 100.]


train.date = train.date.apply(lambda x: dt.datetime.strptime(x, '%d.%m.%Y'))
train.date = train.date.apply(lambda x: dt.datetime.strftime(x,'%Y-%m'))

data = train.groupby(['date','item_id','shop_id']).sum().reset_index()
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

data = label.fit_transform(data['date'].values)
print(data)
