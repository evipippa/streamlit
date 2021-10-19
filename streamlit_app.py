# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:47:15 2021

@author: Evngelia Pippa
"""

import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics


def introduction():
    st.title("**Welcome to Spotawheel!")
    st.subheader(
        """
        Building and Testing Spotawheel's Car Pricing Model!
        """)

    
def xgboost_model(X_train, y_train, X_test, y_test):
    gb_reg = GradientBoostingRegressor()
    gb_reg.fit(X_train, y_train)
    y_pred= gb_reg.predict(X_test)
    st.write("Accuracy on Traing set: " + str(gb_reg.score(X_train,y_train)) )
    st.write("Accuracy on Testing set: " + str(gb_reg.score(X_test,y_test)))
    st.write("\t\t Gradient Boosting Regressor Error Table")
    st.write('Mean Absolute Error      : ' + str( metrics.mean_absolute_error(y_test, y_pred)))
    st.write('Mean Squared  Error      : ' + str( metrics.mean_squared_error(y_test, y_pred)))
    st.write('Root Mean Squared  Error : '+ str( np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
    st.write('R Squared Error          : '+ str( metrics.r2_score(y_test, y_pred)))
    return y_pred

st.set_page_config(
    page_title="Spotawheel", layout="wide", page_icon="./images/flask.png"
)



def body():
    introduction()
    
    
    X_train_file = st.file_uploader("Upload Training Set:",type=['csv'])
    y_train_file = st.file_uploader("Upload Training Price:",type=['csv'])
    X_test_file = st.file_uploader("Upload Test Set :",type=['csv'])
    y_test_file = st.file_uploader("Upload Actual Price:",type=['csv'])

    
    col1, col2 = st.columns((1, 1))

    with col1:
        if X_train_file is not None:
            file_details = {"FileName":X_train_file.name,"FileType":X_train_file.type,"FileSize":X_train_file.size}
            st.write(file_details)
            X_train = pd.read_csv(X_train_file)
            st.dataframe(X_train)
        #else:
        #    X_train = pd.read_csv(r'.\data\X_train_selected.csv')

        if y_train_file is not None:
            y_tr = pd.read_csv(y_train_file)
            y_train = y_tr['Price']
        #else:
        #    y_tr = pd.read_csv(r'.\data\y_train.csv')
        #    y_train = y_tr['Price']

  
                

    with col2:
        if X_test_file is not None:
            file_details = {"FileName":X_test_file.name,"FileType":X_test_file.type,"FileSize":X_test_file.size}
            st.write(file_details)
            X_test = pd.read_csv(X_test_file)
            st.dataframe(X_test)
        #else:
        #    X_test = pd.read_csv(r'.\data\X_test_selected.csv')

        if y_test_file is not None:
            y_tst = pd.read_csv(y_test_file)
            y_test = y_tst['Price']
        #else:
        #    y_tst = pd.read_csv(r'.\data\y_test.csv')
        #    y_test = y_tst['Price']

    if ((X_train_file is not None) and (y_train_file is not None) and (X_test_file is not None) and (y_test_file is not None)):        
        y_pred = xgboost_model(X_train, y_train, X_test, y_test)
       

if __name__ == "__main__":

    body()

