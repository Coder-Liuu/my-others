import numpy as np
from scipy.spatial.distance import pdist
from sklearn.tree import DecisionTreeClassifier
from imblearn.metrics import geometric_mean_score

class Imbanced():
    def __init__(self,K):
        """参数K"""
        self.k = K
        self.estimator = None

    def imbanced_info(self,y_data):
        """查看数据的不平衡比"""
        balance_values = (y_data.value_counts() / y_data.count()).values
        # print("不平衡比是:", balance_values[0]/balance_values[1])
        print("类别分别占比为:")
        print(str(balance_values*100)+'%')

    def make_data(self,x_data,y_data):
        """生成一份去掉负域的数据"""
        # 重置索引
        x_data = x_data.reset_index(drop=True)
        y_data = y_data.reset_index(drop=True)

        # 生成mean的行
        len_data = len(x_data)
        x_data.loc[len_data]  = x_data.mean()

        # 生成距离矩阵
        distance = list()
        data_inv = np.linalg.inv(np.cov(x_data.T))
        # 打印逆协方差矩阵
        # print(data_inv)
        # print(data_inv.shape)

        # 计算距离
        distance = list()
        for i in range(0,len_data+1):
            distance.append(pdist(x_data.loc[[i,len_data]], 'mahalanobis', VI=data_inv).tolist())
        # 添加距离
        x_data['distance'] = np.array(distance)
        # 添加class
        x_data['class'] = y_data
        # 按马氏距离排序
        x_data.sort_values("distance",inplace=True)
        # 删除平均值
        x_data = x_data[1:-1]
        # 重置索引
        x_data = x_data.reset_index(drop=True)
        # 按K分组
        x_data['group'] =  x_data.index.values // (len_data/self.k)
        # 分区域
        data2 = x_data.groupby('group')['class'].agg([min,max])
        '''# {{{
        # 正域
        data2[data2['max'] == 1].index
        # 负域
        data2[data2['min'] == 2].index
        # 不确定域
        data2[ (data2['min'] == 1) & (data2['max'] == 2 )].index
        '''# }}}
        # 去掉负域的数据  取反 用data_choose来表示后来色数据
        data_choose = x_data[~x_data['group'].isin( (data2[data2['min'] == 2].index).tolist() )]
        print('有',len((data2[data2['min'] == 2].index).tolist()),'组被去除了')
        data_target = data_choose.pop('class')
        del data_choose['distance']
        del data_choose['group']
        # 使用剔除后的数据，来进行决策树的训练
        return data_choose,data_target

    def fit(self,x_data,y_data):
        """使用决策树进行训练"""
        self.estimator = DecisionTreeClassifier()
        self.estimator.fit(x_data,y_data)
        print("训练成功")

    def predict(self,x_pred):
        """使用决策树进行预测"""
        return self.estimator.predict(x_pred)

    def g_mean_score(self,y_true,y_pred):
        """查看G-mean的得分"""
        score = geometric_mean_score(y_true,y_pred)
        print('G-mean的得分是',score)
        return score

    def score(self,x_data,y_data):
        score = self.estimator.score(x_data,y_data)
        print("Score的得分是",score)

