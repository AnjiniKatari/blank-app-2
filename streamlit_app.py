## Step 00 - Import of the packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import sklearn as skl

# from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(
    page_title="Califronia Housing Dashboard 🏡",
    layout="centered",
    page_icon="🏡",
)


## Step 01 - Setup
st.sidebar.title("California - Real Estate Agency 🏡")
page = st.sidebar.selectbox("Select Page",["reg code","Visualization 📊", "Automated Report 📑","Prediction"])


#st.video("video.mp4")

st.image("house2.png")

st.write("   ")
st.write("   ")
st.write("   ")
df = pd.read_csv("housing.csv")


## Step 02 - Load dataset
if page == "reg code":

    st.subheader(" Linear Regression Code ")

    

elif page == "Visualization 📊":

    ## Step 03 - Data Viz
    st.subheader("02 Data Viz")

    # col_x = st.selectbox("Select X-axis variable",df.columns,index=0)
    # col_y = st.selectbox("Select Y-axis variable",df.columns,index=1)

    # tab1, tab2, tab3 = st.tabs(["Bar Chart 📊","Line Chart 📈","Correlation Heatmap 🔥"])

    with tab1:
        st.subheader("Bar Chart")
        # st.bar_chart(df[[col_x,col_y]].sort_values(by=col_x),use_container_width=True)

    with tab2:
        st.subheader("Line Chart")
        # st.line_chart(df[[col_x,col_y]].sort_values(by=col_x),use_container_width=True)


    with tab3:
        st.subheader("Correlation Matrix")
        # df_numeric = df.select_dtypes(include=np.number)

        # fig_corr, ax_corr = plt.subplots(figsize=(18,14))
        # # create the plot, in this case with seaborn 
        # sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='coolwarm')
        # ## render the plot in streamlit 
        # st.pyplot(fig_corr)

elif page == "Automated Report 📑":
    st.subheader("03 Automated Report")
    # if st.button("Generate Report"):
    #     with st.spinner("Generating report..."):
    #         profile = ProfileReport(df,title="California Housing Report",explorative=True,minimal=True)
    #         st_profile_report(profile)

    #     export = profile.to_html()
    #     st.download_button(label="📥 Download full Report",data=export,file_name="california_housing_report.html",mime='text/html')


elif page == "Prediction":
    st.subheader("04 Prediction with Linear Regression")
    # df2 = pd.read_csv("housing.csv")
    # ## Data Preprocessing

    # ### removing missing values 
    # df2 = df2.dropna()

    # ### Label Encoder to change text categories into number categories
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()

    # df2["ocean_proximity"] = le.fit_transform(df2["ocean_proximity"])

    # list_var = list(df2.columns)

    # features_selection = st.sidebar.multiselect("Select Features (X)",list_var,default=list_var)
    # target_selection  = st.sidebar.selectbox("Select target variable (Y))",list_var)
    # selected_metrics = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)","Mean Absolute Error (MAE)","R² Score"],default=["Mean Absolute Error (MAE)"])

    # ### i) X and y
    # X = df2[features_selection]
    # y = df2[target_selection]

    # st.dataframe(X.head())
    # st.dataframe(y.head())

    ### ii) train_test_split
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


    # ## Model 

    # ### i) Definition model
    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression()

    # ### ii) Training model
    # model.fit(X_train,y_train)

    # ### iii) Prediction
    # predictions = model.predict(X_test)

    # ### iv) Evaluation 
    # from sklearn import metrics 
    # if "Mean Squared Error (MSE)" in selected_metrics:
    #     mse = metrics.mean_squared_error(y_test, predictions)
    #     st.write(f"- **MSE** {mse:,.2f}")
    # if "Mean Absolute Error (MAE)" in selected_metrics:
    #     mae = metrics.mean_absolute_error(y_test, predictions)
    #     st.write(f"- **MAE** {mae:,.2f}")
    # if "R² Score" in selected_metrics:
    #     r2 = metrics.r2_score(y_test, predictions)
    #     st.write(f"- **R2** {r2:,.3f}")

    # st.success(f"My model performance is of ${np.round(mae,2)}")

    # fig, ax = plt.subplots()
    # ax.scatter(y_test,predictions,alpha=0.5)
    # ax.plot([y_test.min(),y_test.max()],
    #        [y_test.min(),y_test.max() ],"--r",linewidth=2)
    # ax.set_xlabel("Actual")
    # ax.set_xlabel("Predicted")
    # ax.set_title("Actual vs Predicted")
    # st.pyplot(fig)
