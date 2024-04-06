import json
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

#v############ BASIC #############

def isfloat(string):
    try:
        # float() is a built-in function
        float(string)
        return True
    except ValueError:
        return False

def fMap(value):
    x1, x2, y1, y2 = (-1,1, 0, 1)
    x = ((value - x1) * (y2 - y1) / (x2 - x1)) + y1
    return x

def Map(value):
    return int(fMap(value))     
#^############ BASIC #############





#v############ CONVERT #############

def class2grade(theClass):
    print(theClass)
    theClass = theClass.split("_")[1]
    if theClass == "1": return 1
    elif theClass == "0": return 0

#v############ MAIN #############

def cor(filePath):
    #plot correlation
    json2csv(filePath, corBool= True)

def confusion():
    from xgb import XGB
    xgb = XGB(model_path="model_u2net.pkl",data_path="metadata_reg.csv")
    model, truth, pred = xgb.eval()

    thres = [0.6, 0.3] #acc, rej

    #Create the NumPy array for actual and predicted labels.
    truthList = []
    for x in truth:
        if x < thres[1]: truthList.append("Reject")
        elif x < thres[0]: truthList.append("Partial")
        else: truthList.append("Accept")
    
    predList = []
    for x in pred:
        if x < thres[1]: predList.append("Reject")
        elif x < thres[0]: predList.append("Partial")
        else: predList.append("Accept")

    truth = np.array(truthList)
    pred = np.array(predList)
    
    #compute the confusion matrix.
    cm = confusion_matrix(truth, pred) #x, y
    
    #Plot the confusion matrix.
    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=["Accept","Partial","Reject"],
                yticklabels=["Accept","Partial","Reject"])
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.show()


############# MAIN #############

def json2csv(filePath, theType ="reg", corBool = False):
    file = open(filePath)
    data = json.load(file)
    DF = {"r":[],"g":[],"b":[], "rg":[],"rb":[],"bg":[], "r2":[],"g2":[],"b2":[], "rg2":[],"rb2":[],"bg2":[], "grade":[]}
    # DF = {"b":[],"g":[],"r":[], "rg":[],"rb":[],"bg":[], "grade":[]}
    for i, key in enumerate(data.keys()):
        grade = class2grade(key)
        for value, feature in zip(data[key].split(' '), DF.keys()):
            DF[feature].append(float(value))
        DF["grade"].append(grade)
    df = pd.DataFrame(DF)
    if corBool:
        f = pyplot.figure(figsize=(19, 15))
        pyplot.matshow(df.corr(), fignum=f.number)
        pyplot.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
        pyplot.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
        cb = pyplot.colorbar()
        cb.ax.tick_params(labelsize=14)
        pyplot.title('Correlation Matrix', fontsize=16);
        # pyplot.matshow(df.corr())
        pyplot.show()
    else:
        df.to_csv(f"{filePath.split('.')[0]}_{theType}.csv")

#^############ MAIN #############



if __name__ == "__main__":
    json2csv("metadata.json", "reg")
    # cor("metadata.json")
    #confusion()
