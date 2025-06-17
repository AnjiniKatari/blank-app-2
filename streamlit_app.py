## Step 00 - Import of the packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import sklearn as skl

# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

st.set_page_config(
    page_title="Student Scores Analysis",
    layout="centered",
)


## Step 01 - Setup
st.sidebar.title("Student Scores Analysis")
page = st.sidebar.selectbox("Select Page",["About the Data","Visualization 📊", "Automated Report 📑","Prediction"])



# st.image("house2.png")

st.write("   ")
st.write("   ")
st.write("   ")
df = pd.read_csv("student-scores.csv")
score_columns = ["math_score", "biology_score", "english_score", "history_score", "physics_score", "chemistry_score", "geography_score"]
df["average_score"] = df[score_columns].mean(axis=1).round(0)


## Step 02 - Load dataset
if page == "About the Data":

    st.subheader(" About the Data ")
    st.write("This data is comprised of 17 factors: Student ID, First name, Last name, email, " \
    "gender, whether the student has a part time job, how many absences they have, if they do " \
    "extra cirriculars, how many hours they study at home, what their career aspirations are, and " \
    "scores for the following subjects: math, hitsory, english, geography, physics, chemistry and " \
    "biology. ")

    st.write("For the sake of this project, we add a column that computes the total average score for each student. This was acheived by " \
    "adding the scores for each subject (of which there are 7) and divided by 7.")
    
    st.write("For the Linear Regression, the total score was computed and predicted, as opposed to the student's indivdual average.")
    

elif page == "Visualization 📊":

    ## Step 03 - Data Viz
    st.subheader("Visualizing the Data")
    filtds = df.drop(columns=["id", "first_name", "last_name", "email"])
    scores = df.drop(columns=["id", "first_name", "last_name", "email", "gender", "absence_days", "weekly_self_study_hours", "extracurricular_activities", "career_aspiration", "part_time_job"])

    col_x = st.selectbox("Select X-axis variable (group by)", filtds.columns)
    col_y = st.selectbox("Select Y-axis variable (numeric)", scores.columns)

    tab1, tab2, tab3, tab4 = st.tabs(["Box plot", "Bar Chart 📊","Line Chart 📈","Correlation Heatmap 🔥",])

    with tab1:
        st.subheader("Box plot")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=col_x, y=col_y, ax=ax)
        ax.set_title(f"{col_y} by {col_x}")
        st.pyplot(fig)

    with tab2:
        st.subheader("Bar Chart")
        st.bar_chart(df[[col_x,col_y]].sort_values(by=col_x),use_container_width=True)

    with tab3:
        st.subheader("Line Chart")
        st.line_chart(df[[col_x,col_y]].sort_values(by=col_x),use_container_width=True)


    with tab4:
        st.subheader("Correlation Matrix")
        df_numeric = df.select_dtypes(include=np.number)

        fig_corr, ax_corr = plt.subplots(figsize=(18,14))
        # # create the plot, in this case with seaborn 
        sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='coolwarm')
        # ## render the plot in streamlit 
        st.pyplot(fig_corr)

elif page == "Automated Report 📑":
    st.subheader("03 Automated Report")
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            profile = ProfileReport(df,title="California Housing Report",explorative=True,minimal=True)
            st_profile_report(profile)

        export = profile.to_html()
        st.download_button(label="📥 Download full Report",data=export,file_name="california_housing_report.html",mime='text/html')


elif page == "Prediction":
    st.subheader("04 Prediction with Linear Regression")
    df2 = pd.read_csv("student-scores.csv")
    # ## Data Preprocessing

    # ### removing missing values 
    df2 = df2.dropna()
    df2 = df2.drop(columns=['first_name', 'last_name', 'email', 'id', 'math_score', 'english_score', 'physics_score', 'chemistry_score', 'biology_score', 'history_score', 'geography_score', 'gender', 'career_aspiration'])

    # ### Label Encoder to change text categories into number categories
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    #df2["gender"] = le.fit_transform(df2["gender"])
    #df2["part_time_job"] = le.fit_transform(df2["part_time_job"])
    #df2["extracurricular_activities"] = le.fit_transform(df2["extracurricular_activities"])
    #df2["career_aspiration"] = le.fit_transform(df2["career_aspiration"])



    list_var = list(df2.columns)

    features_selection = st.sidebar.multiselect("Select Features (X)",list_var,default=list_var)
    selected_metrics = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)","Mean Absolute Error (MAE)","R² Score"],default=["Mean Absolute Error (MAE)"])

    # ### i) X and y
    X = df2[features_selection]
    df2["average_score"] = df[score_columns].mean(axis=1).round(0)
    y = df2['average_score']

    st.dataframe(X.head())
    st.dataframe(y.head())

    ### ii) train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


    # ## Model 

    # ### i) Definition model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()

    # ### ii) Training model
    model.fit(X_train,y_train)

    # ### iii) Prediction
    predictions = model.predict(X_test)

    # ### iv) Evaluation 
    from sklearn import metrics 
    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE** {mse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE** {mae:,.2f}")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R2** {r2:,.3f}")

    st.success(f"My model performance is of {np.round(mae,2)}")


    fig, ax = plt.subplots()
    ax.scatter(y_test,predictions,alpha=0.5)
    ax.plot([y_test.min(),y_test.max()],
           [y_test.min(),y_test.max() ],"--r",linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
