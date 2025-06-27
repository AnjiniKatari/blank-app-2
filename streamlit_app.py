## Step 00 - Import of the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# MLflow and DagsHub initialization
import mlflow
import mlflow.sklearn
import dagshub
import shap

# Initialize DagsHub with MLflow integration
dagshub.init(repo_owner='jheelkshirin', repo_name='Final', mlflow=True)

#from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics


st.set_page_config(
    page_title="Student Scores Analysis",
    layout="centered",
)


## Step 01 - Setup
st.sidebar.title("Student Scores Analysis")
page = st.sidebar.selectbox("Select Page",["About the Data 🧮","Visualization 📊", "Prediction 🔮", "Explainability 🔍", 'MLflow Runs 🏃‍♂️'])
st.image("schoolpic.jpg")



# st.image("house2.png")

st.write("   ")
st.write("   ")
st.write("   ")
df = pd.read_csv("student-scores.csv")
score_columns = ["math_score", "biology_score", "english_score", "history_score", "physics_score", "chemistry_score", "geography_score"]
df["average_score"] = df[score_columns].mean(axis=1).round(0)


## Step 02 - Load dataset
if page == "About the Data 🧮":

    st.subheader(" About the Data ")
    st.write("This data is comprised of 17 factors: Student ID, First name, Last name, email, " \
    "gender, whether the student has a part time job, how many absences they have, if they do " \
    "extra cirriculars, how many hours they study at home, what their career aspirations are, and " \
    "scores for the following subjects: math, hitsory, english, geography, physics, chemistry and " \
    "biology. ")

    st.write("For the sake of this project, we add a column that computes the total average score for each student. This was acheived by " \
    "adding the scores for each subject (of which there are 7) and divided by 7.")
    
    st.write("For the prediction models, the average score was computed and predicted as well.")
    

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



elif page == 'Prediction 🔮':
    st.subheader("Prediction with MLflow Tracking 🤖")
    df2 = pd.read_csv("student-scores.csv")

    # Data preprocessing
    df2 = df2.dropna()
    df2 = df2.drop(columns=['first_name', 'last_name', 'email', 'id', 'math_score', 'english_score', 'physics_score', 'chemistry_score', 'biology_score', 'history_score', 'geography_score', 'gender', 'career_aspiration'])

    # Feature/Target selection
    list_var = list(df2.columns)
    features_selection = st.sidebar.multiselect("Select Features (X)",list_var,default=list_var)
    selected_metrics = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)","Mean Absolute Error (MAE)","R² Score"],default=["Mean Absolute Error (MAE)"])
    #target_selection = st.sidebar.selectbox("Select Target Variable (Y)", list_var, index=list_var.index('average_score') if 'average_score' in list_var else 0)
    target_selection = st.sidebar.selectbox("Select Target Variable (Y)",  'average_score')


    # Model choice
    model_name = st.sidebar.selectbox(
        "Choose Model",
        ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
    )

    # Hyperparameters
    params = {}
    if model_name == "Decision Tree":
        params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_name == "Random Forest":
        params['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 500, 100)
        params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_name == "XGBoost":
        params['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 500, 100)
        params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)

   

    df2["average_score"] = df[score_columns].mean(axis=1).round(0)

    # Prepare data
    X = df2[features_selection]
    y = df2[target_selection]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate model
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(**params, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(**params, random_state=42)
    elif model_name == "XGBoost":
        model = XGBRegressor(objective='reg:squarederror', **params, random_state=42)

    # Train, predict and log with MLflow
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Log metrics
        mse = metrics.mean_squared_error(y_test, predictions)
        mae = metrics.mean_absolute_error(y_test, predictions)
        r2 = metrics.r2_score(y_test, predictions)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

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


    # Display metrics
    #st.write(f"**MSE:** {mse:,.2f}")
    #st.write(f"**MAE:** {mae:,.2f}")
    #st.write(f"**R² Score:** {r2:.3f}")

    # Plot Actual vs Predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


# Explainability Page
elif page == "Explainability":
    st.subheader("06 Explainability")
    # Load built-in California dataset for SHAP
    X_shap, y_shap = shap.datasets.california()
    # Train default XGBoost model for explainability
    model_exp = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_exp.fit(X_shap, y_shap)

    # Create SHAP explainer and values
    explainer = shap.Explainer(model_exp)
    shap_values = explainer(X_shap)

    # SHAP Waterfall Plot for first prediction
    st.markdown("### SHAP Waterfall Plot for First Prediction")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())


    # SHAP Scatter Plot for 'Latitude'
    st.markdown("### SHAP Scatter Plot for 'Latitude'")
    shap.plots.scatter(shap_values[:, "Latitude"], color=shap_values, show=False)
    st.pyplot(plt.gcf())


# Explainability Page
elif page == "Explainability 🔍":
    st.subheader("Explainability")
    # Load built-in California dataset for SHAP
    #X_shap, y_shap = shap.datasets.california()
    # Train default XGBoost model for explainability

    #df = pd.read_csv("your_dataset.csv")  # Replace with your dataset path
    #X_shap = df.drop(columns=["average_score"])  # Replace 'target' with your actual target column name
    #y_shap = df["average_score"]

    df2 = pd.read_csv("student-scores.csv")
    # ## Data Preprocessing

    # ### removing missing values 
    df2 = df2.dropna()
    df2 = df2.drop(columns=['first_name', 'last_name', 'email', 'id', 'math_score', 'english_score', 'physics_score', 'chemistry_score', 'biology_score', 'history_score', 'geography_score', 'gender', 'career_aspiration'])
    df2["average_score"] = df[score_columns].mean(axis=1).round(0)

    X_shap = df2.drop(columns=["average_score"])
    y_shap = df2['average_score']

    model_exp = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_exp.fit(X_shap, y_shap)

    # Create SHAP explainer and values
    explainer = shap.Explainer(model_exp)
    shap_values = explainer(X_shap)
    st.write('' \
    '')

    # SHAP Waterfall Plot for first prediction
    st.markdown("##### SHAP Waterfall Plot for First Prediction")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())

    st.write('' \
    '')
    # SHAP Scatter Plot for 'absence_days'
    st.markdown("##### SHAP Scatter Plot for Absence Days")
    shap.plots.scatter(shap_values[:,"absence_days"], color=shap_values, show=False)
    st.pyplot(plt.gcf())
    st.write('' \
    '')
    # SHAP Scatter Plot for 'weekly_self_study_hours'
    st.markdown("##### SHAP Scatter Plot for 'Weekly Self Study Hours'")
    shap.plots.scatter(shap_values[:,"weekly_self_study_hours"], color=shap_values, show=False)
    st.pyplot(plt.gcf())

elif page == "MLflow Runs 🏃‍♂️":
    st.subheader("MLflow Runs 📈")
    # Fetch runs
    runs = mlflow.search_runs(order_by=["start_time desc"])
    st.dataframe(runs)
    st.markdown(
        "View detailed runs on DagsHub: [JheelKshirin/Final MLflow](https://dagshub.com/jheelkshirin/Final.mlflow)"
    )


elif page == 'Hyperparameter Tuning':
    #import streamlit as st
    #import pandas as pd
    #import streamlit as st
    #import pandas as pd
    #from pycaret.classification import setup as cls_setup, compare_models as cls_compare, finalize_model as cls_finalize, predict_model as cls_predict, pull as cls_pull
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, finalize_model as reg_finalize, predict_model as reg_predict, pull as reg_pull

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error


    #st.title("🎈 Simple Pycaret App")


    password = st.sidebar.text_input("Enter Password",type="password")
    if password != "studentscore":
        st.sidebar.error('Incorrect Password')
        st.stop()
    st.sidebar.success("Access Granted") 


    df = pd.read_csv("student-scores.csv").sample(n=1000)

    target = st.sidebar.selectbox("Select a target variable",df.columns)

    features = st.multiselect("Select features",[c for c in df.columns if c != target],default=[c for c in df.columns if c != target] )

    if not features:
        st.warning("Please select at least one feature")
        st.stop()

    if st.button("Train & Evaluate"):
        model_df = df[features+[target]]
        st.dataframe(model_df.head())

        with st.spinner("Training ..."):
            reg_setup(data=model_df,target=target,session_id=42,html=False)
            best = reg_compare(sort="R2",n_select=1)
            model = reg_finalize(best)
            comparison_df =reg_pull()

        st.success("Training Complete!")


        st.subheader("Model Comparison")
        st.dataframe(comparison_df)


        with st.spinner("Evaluating ... "):
            pred_df = reg_predict(model,model_df)
            actual = pred_df[target]
            predicted = pred_df["Label"] if "Label" in pred_df.columns else pred_df.iloc[:, -1]

            metrics= {}

            metrics["R2"] = r2_score(actual,predicted)
            metrics["MAE"] = mean_absolute_error(actual,predicted) 

        st.success("Evaluation Done!")

        st.subheader("Metrics")

        cols = st.columns(len(metrics))
        for i, (name,val) in enumerate(metrics.items()):
            cols[i].metric(name, f"{val:4f}")
    
        st.subheader("Predictions")
        st.dataframe(pred_df.head(10))


elif page == 'Hyperparameter Tuning':
    import streamlit as st
    import pandas as pd
    import streamlit as st
    import pandas as pd
    from pycaret.classification import setup as cls_setup, compare_models as cls_compare, finalize_model as cls_finalize, predict_model as cls_predict, pull as cls_pull
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, finalize_model as reg_finalize, predict_model as reg_predict, pull as reg_pull

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error


    #st.title("🎈 Simple Pycaret App")


    password = st.sidebar.text_input("Enter Password",type="password")
    if password != "studentscore":
        st.sidebar.error('Incorrect Password')
        st.stop()
    st.sidebar.success("Access Granted") 


    df = pd.read_csv("student-scores.csv").sample(n=1000)

    target = st.sidebar.selectbox("Select a target variable",df.columns)

    features = st.multiselect("Select features",[c for c in df.columns if c != target],default=[c for c in df.columns if c != target] )

    if not features:
        st.warning("Please select at least one feature")
        st.stop()

    if st.button("Train & Evaluate"):
        model_df = df[features+[target]]
        st.dataframe(model_df.head())

        with st.spinner("Training ..."):
            reg_setup(data=model_df,target=target,session_id=42,html=False)
            best = reg_compare(sort="R2",n_select=1)
            model = reg_finalize(best)
            comparison_df =reg_pull()

        st.success("Training Complete!")


        st.subheader("Model Comparison")
        st.dataframe(comparison_df)


        with st.spinner("Evaluating ... "):
            pred_df = reg_predict(model,model_df)
            actual = pred_df[target]
            predicted = pred_df["Label"] if "Label" in pred_df.columns else pred_df.iloc[:, -1]

            metrics= {}

            metrics["R2"] = r2_score(actual,predicted)
            metrics["MAE"] = mean_absolute_error(actual,predicted) 

        st.success("Evaluation Done!")

        st.subheader("Metrics")

        cols = st.columns(len(metrics))
        for i, (name,val) in enumerate(metrics.items()):
            cols[i].metric(name, f"{val:4f}")
    
        st.subheader("Predictions")
        st.dataframe(pred_df.head(10))
