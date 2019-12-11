import pytest

import numpy as np
import pandas as pd
import gower

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