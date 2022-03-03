import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.title("ABC Inc.")
st.write("House Price Prediction ")
st.image('data//ho.jpg')

data = pd.read_csv('data//train.csv')
DT = pickle.load(open('DT.pkl', 'rb'))
RFR = pickle.load(open('RFR.pkl', 'rb'))

nav = st.sidebar.radio('Navigation', ['Home', 'Prediction', "Visualization"])
if nav == 'Home':
    st.title('ABC Corporation')
    st.subheader('House Price Prediction')
    if st.checkbox('Show Data'):
        st.dataframe(data)
if nav == 'Prediction':
    st.subheader('Please give the following information:')

    POSTED_BY = st.number_input("Base on Posting Builder = 1, Owner = 2, Dealer = 3")

    UNDER_CONSTRUCTION = st.number_input("If it is under construction YES = 1, NO = 0")

    RERA = st.number_input("Is House is  approved By RealEstate Regulation Authority YES = 1, NO = 0")

    BHK_NO = st.number_input("please mention what number of bhk is house", min_value=0.0)

    SQUARE_FT = st.number_input("Enter the Square feet of the house ")

    READY_TO_MOVE = st.number_input("Is  house ready to move YES = 1, NO = 0")

    RESALE = st.number_input("Is the Property is  Resale Property YES = 1, NO = 0")

    BHK = st.number_input("if the house has BHK Bedroom Hall Kitchen YES = 1, NO = 0")

    RK = st.number_input("if the house has RK Room Kitchen  YES = 1, NO = 0")

    x = np.array([POSTED_BY, UNDER_CONSTRUCTION, RERA, BHK_NO, SQUARE_FT, READY_TO_MOVE, RESALE, BHK, RK])

    x = x.reshape(1, 9)

    st.header("Select the REGRESSOR Algorithms")
    Cls = st.sidebar.radio('Algorithms', ['DecisionTreeRegressor', 'RandomForestRegressor', "ALL"])
    if Cls == "DecisionTreeRegressor":
        if st.button('Predict'):
            st.write("DECISION TREE REGRESSOR")
            S = st.title('The predicted House Price ' + str(round(int(DT.predict(x)[0]))) + '₹ in lacs')

    if Cls == "RandomForestRegressor":
        if st.button('Predict'):
            st.write("RANDOM FOREST REGRESSOR")
            S = st.title(f'The predicted House Price  is ' + str(round(int(RFR.predict(x)[0]))) + '₹ in lacs')

    if Cls == "ALL":
        if st.button('Predict'):
            st.title(
                f'The predicted House Price using Decision Tree Regressor ' + str(round(int(DT.predict(x)[0]))) + '₹')
            st.title(
                f'The predicted House Price using RANDOM FOREST REGRESSOR  is' + str(
                    round(int(RFR.predict(x)[0]))) + '₹')

if nav == "Visualization":

