import pandas as pd
test="C:\\Users\\izuma/test.tsv"
train="C:\\Users\\izuma/train.tsv"
test_data=pd.read_csv(test,sep="\t")
train_data=pd.read_csv(train,sep="\t")
test_data=test_data.dropna()
train_data=train_data.dropna()

#test_data["horsepower"]=test_data["horsepower"].apply(lambda x: 1 if x=="?" else 0)
#train_data["horsepower"]=train_data["horsepower"].apply(lambda x: 1 if x=="?" else 0)


train_X = train_data.drop(["id","mpg","horsepower","car name"],axis=1)
train_y = train_data["mpg"]

val_X = test_data.drop(["id","horsepower","car name"],axis=1)

train_y=train_y.astype('int64')
print(train_y.dtype)#dtypeはdetaの確認
from sklearn.linear_model import LinearRegression as LR  #線形回帰モデル
LR_model=LR()
LR_model.fit(train_X,train_y)
val_predictions=LR_model.predict(val_X)
print(LR_model.score(train_X,train_y))
from sklearn.ensemble import RandomForestClassifier as RF
RF_model=RF(n_estimators=1000,random_state=0)
RF_model.fit(train_X,train_y)
A_val_predictions=RF_model.predict(val_X)
print(RF_model.score(train_X,train_y))
from sklearn.ensemble import GradientBoostingClassifier as GB
GB_model=GB(random_state=0,learning_rate=0.01)
GB_model.fit(train_X,train_y)
val_predictions=GB_model.predict(val_X)
print(GB_model.score(train_X,train_y))
test_data["mpg"]=A_val_predictions
A_test=test_data[["id","mpg"]]
A_test.to_csv("sample_submit.csv",index=False,header=False,encoding='cp932')