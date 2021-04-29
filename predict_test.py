import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# データフレームを作成する。
df = pd.DataFrame()

for i in range(4):
    temp_df = pd.read_excel('sample_04-09.xlsx', sheet_name=i, index_col=0)
    df =  pd.concat([df, temp_df], axis=0)

# 作成したデータフレームの形状を確認する。
df.shape

# 日数と合っているか確認する。
from datetime import date

dates = date(2019, 10, 1) - date(2019, 4, 1)
dates

# 欠損値(NaN)を確認する。
df.isnull().head()

df.isnull().sum(axis=0)

# 欠損値の補間方法

df.fillna(df.mean())

df.fillna(df.median())

df.fillna(df.mode().iloc[:, 0])

df.fillna(method='ffill')

df.fillna(method='bfill')

df.fillna(0)

# 欠損値を削除する。
df.dropna(how='any',  inplace=True)

# 0の件数を数える。
df[df.loc[:, :] == 0].count()

# 0意外のデータを取り出す。
df = df.query('積載重量 != 0 and 積載容量 != 0')

# 相関係数を確認する。
df.corr()

# 散布図を描画する。
fig, ax = plt.subplots()

x = df.loc[:, '積載容量']
y = df.loc[:, '走行台数']

ax.scatter(x, y)
ax.set_xlabel('積載容量')
ax.set_ylabel('走行台数')
ax.set_title('説明変数と目的変数の相関')
ax.grid()

plt.show()

# train_test_splitを使ってデータを分割する。
from sklearn.model_selection import train_test_split

X = df[['積載容量']]
y = df['走行台数']

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.3, random_state=0
                                    )

# 説明変数と目的変数の形状を確認する。
print(X.shape, y.shape)

# 学習モデルを作成する。
from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR_model = LR.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)

# r2_scoreで決定係数を確認する。
from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred))

# モデルのメソッドで決定係数を確認する。
print(LR.score(X_train, y_train), '：', LR.score(X_test, y_test))

# 散布図を描画する。
fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)

plt.show()

# データフレームにテストデータと予測データを反映し、差異を確認してみる。
check_df = pd.DataFrame({'test': y_test, 'predict': y_pred})
check_df.sort_values('日付', inplace=True)

# 予測したいExcelファイルを読み込んでデータを学習モデルに与え、予測結果を得る。
predict_df = pd.read_excel('test_10.xlsx', index_col=0)
predict_df.isnull().sum()

pred_X = predict_df[['積載容量']]

result = LR.predict(pred_X)

predict_df.loc[:, '予測走行台数'] = result

# 元々あった走行台数の列を削除する。
# 予測結果の小数点以下を丸める。
predict_df.drop('走行台数', axis=1, inplace=True)
predict_df.loc[:, '予測走行台数'] = predict_df.loc[:, '予測走行台数'].round()

# Excelファイルに保存する。
predict_df.to_excel('result_10.xlsx')

# 重回帰予測を実装する。
# 説明変数を取得する部分を除き、単回帰予測と同様の手順。
X = df[['積載重量', '積載容量']]
y = df['走行台数']

X_train, X_test, y_train, y_test = train_test_split(
                                   X, y, test_size=0.3, random_state=0
                                   )

LR = LinearRegression()
LR_model = LR.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)

r2_score(y_test, y_pred)

predict_df = pd.read_excel('test_10.xlsx', index_col=0)
pred_X = predict_df[['積載重量', '積載容量']]

result = LR.predict(pred_X)

predict_df.loc[:, '予測走行台数'] = result
predict_df.drop('走行台数', axis=1, inplace=True)
predict_df.loc[:, '予測走行台数'] = predict_df.loc[:, '予測走行台数'].round()

# 1つのデータフレームから特定の区間を取り出してデータフレームを作成する。
tokyo_toyama = df.query('区間 == "東京-富山"')
niigata_tokyo = df[df['区間'] == '新潟-東京']

# 相関係数を確認する。
tokyo_toyama.corr()

# 散布図を描画する。
fig, ax = plt.subplots()

ax.scatter(tokyo_toyama.loc[:, '積載容量'], tokyo_toyama.loc[:, '走行台数'])
ax.set_xlabel('積載容量')
ax.set_ylabel('走行台数')
ax.set_title('東京-富山')

plt.show()

# 学習モデルを用意する。
from sklearn.model_selection import train_test_split

X = tokyo_toyama[['積載容量']]
y = tokyo_toyama['走行台数']

X_train, X_test, y_train,  y_test = train_test_split(
                                    X, y, test_size=0.3, random_state=0 
                                    )

# RandomForestRegressorを使ってみる。
from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor()

RFR_model = RFR.fit(X_train, y_train)
y_pred = RFR.predict(X_test)

# 散布図を描画する。
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel('テストデータ(y_test)')
ax.set_ylabel('予測データ(y_pred)')
ax.set_title('結果確認')

plt.show()

# 決定係数を確認する。
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)

# 相関係数を確認する。
import numpy as np

np.corrcoef(y_test, y_pred)

corr_score = np.corrcoef(y_test, y_pred)[0][1]
corr_score

# 相関係数をグラフに反映する。
corr_score = '{:.2}'.format(corr_score)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel('テストデータ(y_test)')
ax.set_ylabel('予測データ(y_pred)')
ax.set_title('相関係数：' + corr_score)

plt.show()

# 未知のデータを予測する。
predict_df = pd.read_excel('test_10.xlsx', sheet_name='東京-富山', index_col=0)

X = predict_df[['積載容量']]

result = RFR_model.predict(X)

predict_df['予測台数'] = result.round()
predict_df.drop('走行台数', axis=1, inplace=True)

# ラベルのエンコーディング（サンプル）
sample = pd.DataFrame({'記号': ['A', 'B', 'C'], '数値': [1, 2, 3,]})
sample_2 = pd.get_dummies(sample, columns=['記号'])

sample_3 = pd.get_dummies(sample.loc[:, '記号'], prefix='記号')

# 曜日列をエンコーディングしてみる。
niigata_tokyo = pd.get_dummies(niigata_tokyo, columns=['曜日'])

# カラム名を取得する。
niigata_tokyo.columns

# 列を並び替える。 
niigata_tokyo = niigata_tokyo.reindex(columns=['区間', 
                                               '曜日_月', '曜日_火', '曜日_水',
                                               '曜日_木', '曜日_金', '曜日_土', '曜日_日',
                                               '積載重量', '積載容量', '走行台数'])

# 学習モデルを作成する。
from sklearn.model_selection import train_test_split

X = niigata_tokyo[['曜日_月', '曜日_火', '曜日_水', '曜日_木', '曜日_金', '曜日_土', '曜日_日', 
                   '積載重量', '積載容量',]]
y = niigata_tokyo['走行台数']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 学習モデルにデータを与え、予測モデルを作成する。
from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR_model = LR.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)

from sklearn.metrics import r2_score

r2_score(y_test, y_pred)

predict_df = pd.read_excel('test_10.xlsx', sheet_name='新潟-東京', index_col=0)

predict_df2 = pd.get_dummies(predict_df, columns=['曜日'])
predict_df2 = predict_df2.reindex(columns=['区間', 
                                           '曜日_月', '曜日_火', '曜日_水',
                                           '曜日_木', '曜日_金', '曜日_土', '曜日_日',
                                           '積載重量', '積載容量', '走行台数'])

X = predict_df2[['曜日_月', '曜日_火', '曜日_水', '曜日_木', '曜日_金', '曜日_土', '曜日_日', 
                '積載重量', '積載容量',]]

result = LR_model.predict(X)

predict_df.loc[:, '予測結果'] = result.round()
predict_df.drop('走行台数', axis=1, inplace=True)
predict_df.to_excel('10月(新潟-東京)_予測.xlsx')