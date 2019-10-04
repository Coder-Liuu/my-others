import pandas as pd
from Aimbance import Imbanced
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from imblearn.metrics import geometric_mean_score

# Imbanced
ans = list()
for i in range(5,6):
    score = list()
    for j in range(20):
        estimator = Imbanced(i*100)
        x_data = pd.read_csv("./abalone19-5-1tra(1).csv")
        y_data = x_data.pop('class')
        # 划分数据集
        x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.2)
        # 查看数据的不平衡比
        print("升级前的不平衡比是:",end='--->')
        estimator.imbanced_info(y_train)
        x_df ,y_df = estimator.make_data(x_train,y_train)
        print("升级后的不平衡比是:",end='--->')
        estimator.imbanced_info(y_df)
        estimator.fit(x_df,y_df)
        y_pred = estimator.predict(x_test)
        score.append(estimator.g_mean_score(y_test,y_pred))
        estimator.score(x_test,y_test)
        print('----------------------------')
    ans.append(sum(score)/20)
print(ans)

""" 
#简单的使用决策
ans = list()
for i in range(1,2):
    score = list()
    for j in range(1,20):
        estimator = DecisionTreeRegressor()
        x_data = pd.read_csv("./abalone19-5-1tra(1).csv")
        y_data = x_data.pop('class')
        # 划分数据集
        x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.2)
        estimator.fit(x_train,y_train)
        y_pred = estimator.predict(x_test)
        score1 = geometric_mean_score(y_test,y_pred)
        score.append(score1)
    ans.append(sum(score)/20)
print(ans)
"""
