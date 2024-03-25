# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:18:27 2024

@author: Wilson Li
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def sortdata(X_test, y_test, key_id):
    order = list(range(len(y_test)))
    if key_id == len(X_test[0]):
        order.sort(key = lambda i : y_test[i])
    else:
        order.sort(key = lambda i : X_test[:,key_id][i])
    return X_test[order], y_test[order]
# thereis no use sorting, only for ploting
# X_test, y_test = sortdata(X_test, y_test, key_id = 4)

def plot_eval(y_test, y_pred):
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.plot(x_ax, abs(y_pred - y_test), label="bias")
    plt.title("Poly-Test and predicted data for D_ave")
    plt.xlabel('test data label')
    plt.ylabel('Y value')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()
    

class AutoTrain:
    def __init__(self, csvdata, Labels, output, outputid):
        self.data = pd.read_csv(csvdata, delimiter=',')
        self.Labels = Labels
        self.output = output
        self.outputname = output[outputid]
        self.y = np.array([float(i) for i in self.data[self.outputname]])

    def getCovMat(self):
        datas = np.array(self.data[self.Labels+self.output]).T
        for dts in datas:
            dts -= dts.mean()
            dts /= dts.std()
        return np.cov(datas)
    
    def batchTrain(self, labelid, batchnum = 1, test_size = 0.2, rm_ratio = 0.2):
        n = (len(labelid) + len(labelid)**2) >> 1
        self.X = np.array(self.data[[self.Labels[i] for i in labelid]])
        self.rst = [[] for i in range (6)]
        self.power = []
        # print(labelid)
        for i in range(batchnum):
            if batchnum == 1:
                X_train, X_test, y_train, y_test = self.X, self.X, self.y, self.y
            else:
                X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                                    test_size=test_size, 
                                                                    random_state=i+1)
            # fittin polynomial's indexs
            param_grid = {'polynomialfeatures__degree': np.arange(1, 5)}
            model = make_pipeline(PolynomialFeatures(), LinearRegression())
            # n_jobs should not be changed!
            grid = GridSearchCV(model, param_grid, cv = 5)

            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            linear = grid.best_estimator_.named_steps['linearregression']
            coef = np.array([linear.intercept_,] + linear.coef_[1:].tolist())
            power = grid.best_estimator_.named_steps['polynomialfeatures'].powers_
            self.rst[0].append(r2_score(y_test, y_pred))
            self.rst[1].append(np.sqrt(np.mean((1 - y_pred/y_test)**2)))
            self.rst[2].append(mean_absolute_error(y_test, y_pred))
            self.rst[3].append(mean_squared_error(y_test, y_pred))
            # Calculate the coefficients
            # Drop higher degree while trainning
            if grid.best_params_['polynomialfeatures__degree'] > 2:
                coef = coef[:n + len(labelid) + 1]
            elif grid.best_params_['polynomialfeatures__degree'] < 2:
                if len(self.power) == 0:  self.power = power  # Use the One Degree model at first
                coef = np.pad(coef, (0, n), 'constant', constant_values = (0,0))
            else:  self.power = power    # if Power == 2 then calculate in this
            self.rst[4].append(coef)
            self.rst[5].append(grid)
        
        # rank the models for remove
        if rm_ratio > 0 and rm_ratio < 1:
            rank = [i for i in range(batchnum)]
            rank.sort(key = lambda x : self.rst[0][x])
            for i in range(5):
                self.rst[i] = [self.rst[i][item] for item in rank]
                self.rst[i] = self.rst[i][int(np.floor(batchnum*rm_ratio)):]
                
        return {"labels" : [self.Labels[i] for i in labelid],
                "R² Score" : np.mean(self.rst[0]),
                "Root Mean Squared Error" : np.mean(self.rst[1]),
                "Mean Absolute Error" : np.mean(self.rst[2]),
                "Mean Squared Error" : np.mean(self.rst[3]),
                "Coefficient" : np.mean(self.rst[4], axis = 0),
                "model" : self.GrossModel(self.rst[5])}
                       
    
    def __getCombinations(self, ids):
        combinations = []
        bits = len(ids) # 6
        # Use bin to generate labels
        for i in range(1, 2**bits):
            label = bin(i).rjust(bits,'0')[-bits:]
            tag = []
            for j, ltr in enumerate(label):
                if ltr == '1':
                    tag.append(ids[j])
            combinations.append(tag)
        power = self.Labels.index('Laser Power (W)')
        speed = self.Labels.index('Scanning Speed (mm/s)')
        return [comb + [power, speed] for comb in combinations]
    
    
    def combineFit(self, labelid, batchnum):
        # try to sort with binary
        combinations = self.__getCombinations(labelid)
        # print(combinations)
        results = [self.batchTrain(labelid = comb, batchnum = batchnum, rm_ratio = 0) for comb in combinations]
        results.sort(key = lambda x : x["R² Score"], reverse = True)
        return results
    
    
    class GrossModel:
        def __init__(self, models):
            self.models = models
    
        def predict(self, X_test):
            y_pred = []  # eval on whole set
            for grid in self.models:
                y_pred.append(grid.predict(X_test))
            return np.mean(np.array(y_pred), axis = 0)
    
    
    def fullFit(self, labelid, test_size = 0.2, batchnum = 100, rm_ratio = 0.2, prt = True):
        if test_size == 0:
            result = self.batchTrain(labelid = labelid, test_size = test_size, rm_ratio = rm_ratio)
            self.model = self.rst[5][0]
            if prt: print("On 110 Single trainning:")
        else:
            result = self.batchTrain(labelid = labelid, batchnum = batchnum, test_size = test_size, rm_ratio = rm_ratio)
            if prt: print("On 110 Gross trainning:")
            self.model = self.GrossModel(self.rst[5])
            y_pred = self.model.predict(self.X)
            result["R² Score"] = r2_score(self.y, y_pred)
            result["Root Mean Squared Error"] = np.sqrt(np.mean((1 - y_pred/self.y)**2))
            result["Mean Absolute Error"] = mean_absolute_error(self.y, y_pred)
            result["Mean Squared Error"] = mean_squared_error(self.y, y_pred)
        # calculate the formula
        if prt: 
            self.coeff = result["Coefficient"]
            formula = self.outputname + ' = ' + str(self.coeff[0])
            for i in range(len(self.power)-1):  # Model's Power
                coef = self.coeff[i+1]
                if coef > 0:
                    formula += ' + ' # plus
                else: formula += ' ' # minus
                formula += str(coef)+' '
                for j, pw in enumerate(self.power[i+1]):
                    if pw != 0 :
                        formula += self.Labels[labelid[j]]
                        if pw > 1:
                            formula += '^'+str(pw)+' '
            print(formula)
        return result
    
    def testOn(self, testbatch, labelid):
        testX = np.array(testbatch.data[[self.Labels[i] for i in labelid]])
        y_pred  = self.model.predict(testX)
        return {"R² Score" : r2_score(testbatch.y, y_pred),
                "Root Mean Squared Error": np.sqrt(np.mean((1 - y_pred/testbatch.y)**2)),
                "Mean Absolute Error" : mean_absolute_error(testbatch.y, y_pred),
                "Mean Squared Error" : mean_squared_error(testbatch.y, y_pred)}
    
    def CombineFit(self, testbatch, labelid):
        combinations = self.__getCombinations(labelid)
        results = []
        for comb in combinations:
            modeldict = self.batchTrain(labelid = comb, test_size = 0, batchnum = 1, rm_ratio = 0)
            self.model = self.rst[5][0]
            dic = self.testOn(testbatch, comb)
            dic["labels"] = [self.Labels[i] for i in comb]
            dic["model"] = self.model
            results.append(dic)
        results.sort(key = lambda x : x["R² Score"], reverse = True)
        return results

# Choose model parameters from R² Score added together
def UnitedModel(modelrank, ModelRank, sortkey = "R² Score"):
    models = [{"labels": model["labels"], sortkey.lower() : model[sortkey]} for model in modelrank]
    for model in models:
        for Model in ModelRank:
            if Model["labels"] == model["labels"]:
                model[sortkey] = Model[sortkey]
                model["model"] = Model["model"]
    reverse = True if sortkey == "R² Score" else False
    # Weighted sort
    index = 0.8 if sortkey == "R² Score" else 1.25
    models.sort(key = lambda x: x[sortkey]+x[sortkey.lower()]*index, reverse = reverse)
    # Choose the best models for predictions
    # best_models = [models[0]]
    best_models = []
    
    # can adjust whether to save raw MPM predictions
    Tags = [['OT_Int (0-255)', 'OT_Max (0-255)'],
           # ['MPM_on (0-255)', 'MPM_off (0-255)'],
            ['Calib_on', 'Calib_off']]

    Extra = ['Laser Power (W)', 'Scanning Speed (mm/s)']
    for model in models:
        if model["labels"] in [
               # Tags[0] + Extra, 
                               #[Tags[0][0]]+Extra, 
                               [Tags[0][1]]+Extra,
                               ]:
            best_models.append(model)
            break
    for model in models:
        if model["labels"][0] in Tags[0] and model["labels"][1] in Tags[1]:
            best_models.append(model)
            break
            
    return best_models

# save models in format of .pkl
def SaveModelist(dirct, models):
    for model in models: 
        filename = ''
        for name in model['labels'][:-2]:
            filename += (name+',')
        filename = dirct+filename[:-1] + '.pkl'
        with open(filename,"wb") as f:
            pickle.dump(model, f)

 
if __name__ == "__main__":
    Labels = ['OT_Int (0-255)',         # 0
              'OT_Max (0-255)',         # 1
              'MPM_on (0-255)',         # 2
              'MPM_off (0-255)',        # 3
              'Calib_on',               # 4
              'Calib_off',              # 5
              'Hatch Space (mm)',       # 6
              'Laser Power (W)',        # 7
              'Scanning Speed (mm/s)']  # 8
    labelid = list(range(6))
    output = ['D_mean (µm)', 'W_mean (µm)']
    trainner = AutoTrain('RESULT.csv', Labels, output, outputid = 1)
    trainer = AutoTrain('28Case/28_result.csv', Labels, output, outputid = 1)
    cov2 = trainer.getCovMat()
    cov = trainner.getCovMat()
    modelrank = trainner.combineFit(labelid = labelid, batchnum = 100)
    # result = trainner.fullFit(labelid, test_size = 0, batchnum = 1, rm_ratio = 0.2)
    ModelRank = trainner.CombineFit(trainer, labelid)
    models = UnitedModel(modelrank, ModelRank, sortkey = "R² Score")
    SaveModelist('models/modelsImageW/', models)
