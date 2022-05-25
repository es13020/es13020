import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Working abroad study", page_icon="📊", initial_sidebar_state="expanded"
)

st.write(
    """
# 📊 Working abroad study
Upload your Excel sheet.
"""
)

uploaded_file = st.file_uploader("Upload Excel sheet", type=".xlsx")

use_example_file = st.checkbox(
    "Use example file", False, help="Use in-built example Excel sheet"
)

if use_example_file:
    uploaded_file = "Python data.xlsx"
    xi_default = ['study_abroad', 'female', 'internship', 'father_highschool', 'father_university', 'mother_highschool','mother_university']

if uploaded_file:

    # Reading data
    df = pd.read_excel(uploaded_file)

    st.markdown("### Data preview")

    st.write("#### First 5 rows")
    st.dataframe(df.head())

    st.write("#### Data exploration")
    st.write("##### People studying/working abroad for each university")
    st.bar_chart(df.groupby('university')['work_abroad', 'study_abroad'].sum(), use_container_width=True, height=500)


    # Select Xi variables
    st.markdown("### Select columns will be used to predict 'work_abroad' value")
    with st.form(key="my_xi"):
            variables=list(df.columns)
            variables.remove("work_abroad")
            xi = st.multiselect(
                "Xi columns",
                options=variables,
                help="Select columns will be used as Xi variables.",
                default=xi_default,
            )

            submit_button = st.form_submit_button(label="Submit")
    if xi:
        print(xi)
        X = df[xi]
        print(X.columns)
        y = df.work_abroad

        # Train/Test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Choice of regression model (Logistic Regression)
        reg = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)

        # Model training
        reg.fit(X_train, y_train)

    st.write("## Results for Logistic Regression on data from ", uploaded_file)

    # Model metrics on training data
    print(accuracy_score(y_train, reg.predict(X_train)))
    st.write("##### Metrics on training data :")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.metric("Accuracy", value=f"{100*accuracy_score(y_train, reg.predict(X_train)):.7g}%",)
    with mcol2:
        st.metric("Mean squared error", value=f"{mean_squared_error(y_train, reg.predict(X_train)):.7g}")

    # Model metrics on test data
    st.write("##### Metrics on test data :")
    print(accuracy_score(y_test, reg.predict(X_test)))
    mcol3, mcol4 = st.columns(2)
    with mcol3:
        st.metric("Accuracy", value=f"{100*accuracy_score(y_test, reg.predict(X_test)):.7g}%",)
    with mcol4:
        st.metric("Mean squared error", value=f"{mean_squared_error(y_test, reg.predict(X_test)):.7g}")

    st.write("##### Model intercepts and coefficients :")
    coefficients = [list(reg.intercept_) + list(reg.coef_[0])] 
    coef_names = ["intercept"] + xi
    df2 = pd.DataFrame(coefficients, columns = coef_names) 
    st.dataframe(df2)

    st.write("##### Model classes :")
    colnames = [f"Class {i}" for i in range(1,len(reg.classes_)+1)] 
    df3 = pd.DataFrame([reg.classes_] , columns = colnames) 
    st.dataframe(df3)

    st.write("##### Predicted probabilities on test data :")
    df4 = pd.DataFrame(reg.predict_proba(X_test) , columns = colnames) 
    st.dataframe(df4)
 
