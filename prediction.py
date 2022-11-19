import numpy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
judge=[]
best1=[]
best2=[]
###data read in
RPPA=pd.read_csv("new_RPPA.csv",index_col=0)
dp=pd.read_csv("new_dp.csv",index_col=0)
### train_data
##build baseline model
x_train,x_test,y_train,y_test=train_test_split(RPPA,dp,test_size=0.3,random_state=None)
##linear model
###k fold validation
for i in range(161):
    fit1 = LinearRegression()
    fit2 = RandomForestRegressor(max_depth=5)
    fit3 = tree.DecisionTreeRegressor(max_depth=5)
    x_T=x_train.iloc[:,i]
    y_T=y_train.iloc[:,i]
    kf = KFold(n_splits=10, shuffle=True)
    scores1=[]
    scores2=[]
    scores3=[]
    scores4=[]
    for i in range(10):
        for t_index, v_index in kf.split(x_T):
            x_t = pd.DataFrame(x_T[t_index])
            x_v = pd.DataFrame(x_T[v_index])
            y_t = pd.DataFrame(y_T[t_index])
            y_v = pd.DataFrame(y_T[v_index])
            y_transformed=y_t.values.ravel()
            fit1.fit(x_t,y_t)
            fit2.fit(x_t,y_transformed)
            fit3.fit(x_t,y_transformed)
            y_mean=numpy.mean(x_t)
            y_base=numpy.tile(y_mean,x_v.shape[0])  ####baseline
            y_predict_train_LN=fit1.predict(x_v)
            y_predict_train_RF=fit2.predict(x_v)
            y_predict_train_DT=fit3.predict(x_v)
            score1=np.sqrt(mean_squared_error(y_v,y_predict_train_LN))
            score2=np.sqrt(mean_squared_error(y_v,y_predict_train_RF))
            score3=np.sqrt(mean_squared_error(y_v,y_predict_train_DT))
            score4=np.sqrt(mean_squared_error(y_v,y_base))
            scores1.append(score1)
            scores2.append(score2)
            scores3.append(score3)
            scores4.append(score4)
    score1_mean = numpy.mean(scores1)
    score2_mean = numpy.mean(scores2)
    score3_mean = numpy.mean(scores3)
    score4_mean = numpy.mean(scores4)
    x1 = pd.DataFrame(x_test.iloc[:,i])
    y1 = pd.DataFrame(y_test.iloc[:,i])
    y_mean1 = numpy.mean(x1)
    y_base1 = numpy.tile(y_mean1, 134)
    y1_predict_test_LN = fit1.predict(x1)
    y1_predict_test_RF = fit2.predict(x1)
    y1_predict_test_DT = fit3.predict(x1)
    score_M1 = np.sqrt(mean_squared_error(y1, y1_predict_test_LN))
    score_M2 = np.sqrt(mean_squared_error(y1, y1_predict_test_RF))
    score_M3 = np.sqrt(mean_squared_error(y1, y1_predict_test_DT))
    score_M4 = np.sqrt(mean_squared_error(y1, y_base1))
    score5 = r2_score(y1, y1_predict_test_LN)
    score6 = r2_score(y1, y1_predict_test_RF)
    score7 = r2_score(y1, y1_predict_test_DT)
    score8 = r2_score(y1, y_base1)
    if (score_M4 == min(score_M1, score_M2, score_M3, score_M4)) or (score4_mean == min(score1_mean,score2_mean,score3_mean,score4_mean)):
            best1.append("base")
    else:
        if (score5 == max(score5, score6, score7, score8)):
            best1.append("LN")
        if (score6 == max(score5, score6, score7, score8)):
            best1.append("RF")
        if (score7 == max(score5, score6, score7, score8)):
            best1.append("DT")
print(pd.value_counts(best1))###best among 3 methods
####modify parameter

###have not excluded failed yet because only 2 excluded
''''
importance=[]
coef=[]
R2=[]
data=pd.DataFrame(RPPA)
from scipy.stats import pearsonr
for i in range(161):
    fit4 = LinearRegression()
    x_T = pd.DataFrame(RPPA.iloc[:, i])
    y_T = pd.DataFrame(dp.iloc[:, i])
    fit4.fit(x_T,y_T)
    y_predict = fit4.predict(x_T)
    r1,p1=pearsonr(pd.DataFrame(y_predict).values.ravel(), pd.DataFrame(y_T).values.ravel())
    coef.append(r1)
    R2.append(r2_score(y_T,y_predict))
    data.iloc[:,i]=y_predict
    importance.append(fit4.coef_[0])
coef=pd.DataFrame(coef)
R2=pd.DataFrame(R2)
data=pd.DataFrame(data)
data.to_csv("data.csv")
importance=pd.DataFrame(importance)
importance.to_csv("importance.csv")
R2.to_csv("R2.csv")
coef.to_csv("coef.csv")
####using gridsearchCV for best parameter
best_score=[]
best_para =[]
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
for i in range(161):
    x_T = pd.DataFrame(x_train.iloc[:,i])
    y_T = pd.DataFrame(y_train.iloc[:,i])
    y_transformed = y_T.values.ravel()
    param_test1 = {"n_estimators":range(1,101,10)}
    gsearch1 = GridSearchCV(estimator=RandomForestRegressor(),param_grid=param_test1,scoring='neg_mean_squared_error',cv=10,n_jobs=5)
    gsearch1.fit(x_T,y_transformed)
    best_score.append(gsearch1.best_score_)
    best_para.append(gsearch1.best_params_)
print(best_para)
print(best_score)
'''














