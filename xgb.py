import os
import numpy as np
import time

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd

if __name__ == "__main__" or __name__ == "utils" or __name__ == "xgb" :
    from Pwd import pwd
    from utils import read_csv, fMap
else:
    from .Pwd import pwd
    from .utils import read_csv, fMap



import pickle

class XGB:
    def __init__(self, model_path = None, data_path = None):

        # Give data_path + model_path, this class will train the data and save it in to a model
        # Give just model_path, this class will load the model and return the prediction(s).
        self.model_path = model_path
        self.data_path = data_path

        self.flag = 0 # 0: Done Training/initial, 1: Done Predicting
        
        self.train_dmatrix = None
        self.test_dmatrix = None
        self.test_y = None
        
        self.model_param = {"u2netp":{"param":{"learning_rate":0.1, "booster":"gblinear", "objective":"reg:squarederror"},
                             "num_boost_round":500},
                            "u2net":
                            {"param":{"learning_rate":0.01, "booster":"gblinear", "objective":"reg:squarederror"},
                             "num_boost_round":10000}}
        self.param = self.model_param["u2net"]["param"]
        self.num_boost_round = self.model_param["u2net"]["num_boost_round"]

        
        #0:small size, [1][default]: big size

        if model_path:
            self.model = pickle.load(open(self.model_path, 'rb'))
        else:
            self.model = None

        if data_path:
            data = read_csv(self.data_path)
            X, y = data.iloc[1:, :-1], data.iloc[1:, -1:]
            train_X, test_X, train_y, self.test_y = train_test_split(X, y, 
                            test_size = 0.1, random_state = 123)
            self.train_dmatrix = xgboost.DMatrix(data = train_X, label = train_y)
            self.test_dmatrix = xgboost.DMatrix(data = test_X, label = self.test_y)

    def train(self, model_path = None, data_path = None, quiet = True, show_predict = True):
        if model_path == None:
            if self.model_path:
                model_path = self.model_path
            else:
                print("Please insert model path")
                return
        
        if data_path == None:
            if self.data_path:
                data_path = self.data_path
            else:
                print("Please insert data path")
                return

        start = time.time()
        if not quiet:
            print(f'Start Training "{data_path}"')
            print(self.param, self.num_boost_round)
        model = xgboost.train(params = self.param, dtrain = self.train_dmatrix, num_boost_round = self.num_boost_round)
        
        
        pred = model.predict(self.test_dmatrix)
        if not quiet:
            if show_predict:
                for x, y in zip(self.test_y["grade"],pred):
                    print(x,y)
                rmse = np.sqrt(MSE(self.test_y, pred))
                print("RMSE : % f" %(rmse))

        pickle.dump(model, open(model_path, 'wb'))
        print(f'model saved as "{model_path}"')
        
        self.flag = 0
        print("Done in:",time.time()-start)
        return model, self.test_y["grade"], pred
    
    def eval(self, model_path = None, data_path = None, quiet = False, show_predict = True):
        if model_path == None:
            if self.model_path:
                model_path = self.model_path
            else:
                print("Please insert model path")
                return
        
        if data_path == None:
            if self.data_path:
                data_path = self.data_path
            else:
                print("Please insert data path")
                return

        start = time.time()
        pred = self.model.predict(self.test_dmatrix)
        if not quiet:
            if show_predict:
                for x, y in zip(self.test_y["grade"],pred):
                    print(x,y)
                rmse = np.sqrt(MSE(self.test_y, pred))
                print("RMSE : % f" %(rmse))

        print("Done in:",time.time()-start)
        return self.model, self.test_y["grade"], pred
    
    def predict(self, model_path = None, data = None):
        if type(data) == str:
            data = data.split(' ')
            data = [float(x) for x in data]
        print(data)
        data.insert(0,0)
        data.append(0)
        
        if type(data) != xgboost.DMatrix:
            DF = {"Unnamed: 0":[], "r":[],"g":[],"b":[], "rg":[],"rb":[],"bg":[],"r2":[],"g2":[],"b2":[],"rg2":[],"rb2":[],"bg2":[], "grade":[]}
            for i, key in enumerate(DF.keys()):
                DF[key].append(data[i])
            for i, key in enumerate(DF.keys()):
                DF[key].append(0)
            #del DF["Unnamed: 0"]
            df = pd.DataFrame(DF)
            X, y = df.iloc[:, :-1], df.iloc[:, -1:]
            train_X, test_X, train_y, self.test_y = train_test_split(X, y, 
                            test_size = 0.5, random_state = 2)
            data = xgboost.DMatrix(train_X,train_y)

        if model_path == None:
            if self.model_path:
                model_path = self.model_path
            else:
                print("Please insert model path")
                return
        self.model = pickle.load(open(model_path, 'rb'))
        pred = self.model.predict(data)[0]
        
        self.flag = 1
        # return pred
        return fMap(pred)

    def select_model(self, model_name):
        try:
            self.model = pickle.load(open(os.path.join(pwd(__file__), f"model_{model_name}.pkl"), 'rb'))
            self.model_path = os.path.join(pwd(__file__),f"model_{model_name}.pkl")
            #0:small size, 1: big size
        except Exception as e:
            print(e)
        
        self.param = self.model_param[model_name]["param"]
        self.num_boost_round = self.model_param[model_name]["num_boost_round"]

    def mqtt_publish(self):... # implemented in main

if __name__ == "__main__":
    model_name="u2netp"
    xgb = XGB()
    #xgb.select_model(model_name)
    xgb.train(model_path="model.pkl",data_path = "metadata_reg.csv")
    data = [str(x) for x in [1,1,1,1,0,0,0,1,1,1,0,0,0,0.0]]
    print(xgb.predict(model_path="model.pkl", data= ' '.join(data)))	
