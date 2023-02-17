import pytest

import numpy as np
import pandas as pd
import gower
import time

def test_answer():
    Xd=pd.DataFrame({'age':[21,21,19, 30,21,21,19,30,None],
                     'gender':['M','M','N','M','F','F','F','F',None],
                     'civil_status':['MARRIED','SINGLE','SINGLE','SINGLE','MARRIED','SINGLE','WIDOW','DIVORCED',None],
                     'salary':[3000.0,1200.0 ,32000.0,1800.0 ,2900.0 ,1100.0 ,10000.0,1500.0,None],
                     'has_children':[1,0,1,1,1,0,0,1,None],
                     'available_credit':[2200,100,22000,1100,2000,100,6000,2200,None]})
    Yd = Xd.iloc[1:3,:]
    X = np.asarray(Xd)
    Y = np.asarray(Yd)
    aaa = gower.gower_matrix(X)
    assert aaa[0][1] == pytest.approx(0.3590238, 0.001)
    
    
def test_multiprocessing():
    Xd=pd.DataFrame({'age':[21,21,19, 30,21,21,19,30,None],
                     'gender':['M','M','N','M','F','F','F','F',None],
                     'civil_status':['MARRIED','SINGLE','SINGLE','SINGLE','MARRIED','SINGLE','WIDOW','DIVORCED',None],
                     'salary':[3000.0,1200.0 ,32000.0,1800.0 ,2900.0 ,1100.0 ,10000.0,1500.0,None],
                     'has_children':[1,0,1,1,1,0,0,1,None],
                     'available_credit':[2200,100,22000,1100,2000,100,6000,2200,None]})
    Yd = Xd.iloc[1:3,:]
    X = np.asarray(Xd)
    Y = np.asarray(Yd)
    aaa = gower.gower_matrix(X,n_jobs=-1)
    assert aaa[0][1] == pytest.approx(0.3590238, 0.001)
    
    
    
def test_multiprocessing_large():
    results = []
    for size in np.arange(10000,100000,10000):
        X = np.random.random((size, 8))                                                  
        start_time = time.time()                                                 
        aaa = gower.gower_matrix(X)
        time_single = (time.time() - start_time)
        print("Size: %s Single thread--- %s seconds ---" % (size,time_single))
        start_time = time.time()    
        aaa_multi = gower.gower_matrix(X,n_jobs=-1)
        time_multi = (time.time() - start_time)
        print("Size: %s Multithread --- %s seconds ---" % (size, time_multi))
        assert aaa[0][1] == pytest.approx(aaa_multi[0][1], 0.001)
        assert (sum(sum(aaa-aaa_multi))) == pytest.approx(0, 0.001)
        results.append([size, time_single,time_multi])
    
    return results
    
    
def test_multiprocessing_large_nominal():
    results = []
    for size in np.arange(10000,100000,10000):
        X = np.random.random((size, 4))                                                  
        Xn = np.random.randint(2, size=(X.shape[0],4))
        X = np.concatenate((X,Xn), axis=1)
        start_time = time.time()                                                 
        aaa = gower.gower_matrix(X,cat_features=[False]*4+[True]*4)
        time_single = (time.time() - start_time)
        print("Size: %s Single thread --- %s seconds ---" % (size,time_single))
        start_time = time.time()    
        aaa_multi = gower.gower_matrix(X,n_jobs=-1,cat_features=[False]*4+[True]*4)
        time_multi = (time.time() - start_time)
        print("Size: %s Multithread --- %s seconds ---" % (size, time_multi))
        assert aaa[0][1] == pytest.approx(aaa_multi[0][1], 0.001)
        assert (sum(sum(aaa-aaa_multi))) == pytest.approx(0, 0.001)
        results.append([size, time_single,time_multi])
    
    return results
    
