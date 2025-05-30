import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import metrics

def seconds_to_min_secs(seconds):
    mins = seconds//60
    seconds = seconds % 60
    return(mins,seconds)

st.title('5km Parkrun Time Predictor 🏃')

selected_age = st.select_slider('Select Runner Age:',options=range(10, 101,1),value=30)
selected_n_runs = st.select_slider('Enter # of previous runs:',options=range(0, 1001,1),value=0)
selected_gender = st.selectbox('Select Gender:',options=['Male','Female'])
selected_gender = selected_gender == 'Male'
if st.button("Make Prediction", use_container_width=True):
## Put test features into formatted df
    test_df = pd.DataFrame([[selected_age,selected_n_runs,selected_gender]], columns=['Age Group', 'Runs', 'Gender_Male'])

    ## Read Data from csv

    parkrun = pd.read_csv("Parkrun_data.csv")

    ## Clean and transform data
    parkrun['Time (s)'] = parkrun['Time'].apply(lambda x: x // 1_000_000_000)
    parkrun = parkrun[['Age Group','Gender','Runs','Time (s)']]


    ## Convert columns to numeric
    parkrun = pd.get_dummies(parkrun, columns=['Gender'], drop_first=True)
    parkrun = parkrun[~parkrun['Age Group'].isin(['SW---', 'MWC','WWC', 'SM---'])]
    parkrun['Age Group'] = parkrun['Age Group'].str[2:]

    age_order ={'10':10,'11-14':12,'15-17':16,'18-19':18,'20-24':22,'25-29':27,
            '30-34':32,'35-39':37,'40-44':42,'45-49':47,'50-54':52,'55-59':57,'60-64':62,'65-69':67,'70-74':72,'75-79':77,
            '80-84':82,'85-89':87,'90-94':92,'95-99':97}

    parkrun['Age Group'] = parkrun['Age Group'].map(age_order)
    ## Set target y  and features X
    parkrun_5000 = parkrun.sample(20000,random_state=100)
    y = parkrun_5000['Time (s)']
    X = parkrun_5000[['Age Group', 'Runs', 'Gender_Male']]

    ## Train model
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3,random_state=101)
    lm = LinearRegression()
    lm.fit(X_train,y_train)

    # Make Predictions

    predictions = lm.predict(X_test)

    myentry_poly = poly.fit_transform(test_df)
    prediction = lm.predict(myentry_poly)
    predicted_time_formatted = f"{int(prediction[0] // 60)}:{int(prediction[0] % 60):02d}"
    st.write(f"Predicted Finish Time: :orange-background[{predicted_time_formatted}]")
 
    ## Model Evaluation

    with st.expander("Model Evaluation"):
        MAE = metrics.mean_absolute_error(y_test, predictions)
        MSE = metrics.mean_squared_error(y_test, predictions)
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))
        col1,col2,col3 = st.columns(3)
        with col1:
            st.metric("MAE", (f"{round(MAE)} s"))
        with col2:
            st.metric("MSE", (f"{round(MSE)}"))
        with col3:
            st.metric("RMSE", f"{round(RMSE)} s")
        
        if test_df.iloc[:, -1][0] == 1:
            color = "red"
        else: color = "blue"
        
        ## Predictions vs Actual
        
        plt.scatter(y_test, predictions, s=0.1, c=X_test[:, -1], cmap='coolwarm', alpha=1)
        plt.colorbar(label='Gender (0=Female, 1=Male)')
        plt.ylim(0, predictions.max())
        plt.xlim(0, y_test.max())
        min_val = 0
        max_val = max(y_test.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1)
        plt.scatter(prediction[0],prediction[0], s=100, c=color, edgecolors='black', label='Predicted Value', marker='D')
        plt.xlabel('Actual Finish Time(s)')
        plt.ylabel('Predicted Finish Time(s)')
        
        st.pyplot(plt, use_container_width=False)
        
        ## Residuals plot
        fig, ax = plt.subplots()
        sns.histplot((y_test - predictions), bins=50, ax=ax)
        st.pyplot(fig)
        
    with st.expander(label="How does the size of our dataset impact accuracy?"):
        st.write("Let's see how the size of our dataset impacts the accuracy of our predictions.")
        st.write("Using samples of our dataframe we can create and train multiple models.")
        st.code("""data_range = range(50,5000,50)
        for num_of_data_points in data_range:
        parkrun_filtered = parkrun.sample(num_of_data_points)
        """)
        
        MAE = []
        RMSE = []
        data_range = range(50,5000,50)
        for num_of_data_points in data_range:
            parkrun_filtered = parkrun.sample(num_of_data_points,random_state=100)
            y = parkrun_filtered['Time (s)']
            X = parkrun_filtered[['Age Group', 'Runs', 'Gender_Male']]
            ## Train model
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3,random_state=101)
            lm = LinearRegression()
            lm.fit(X_train,y_train)
            predictions = lm.predict(X_test)
            MAE.append(metrics.mean_absolute_error(y_test, predictions))
            RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

        # Create a new figure
        fig, ax = plt.subplots(figsize=(10,6))

        ax.plot(data_range, MAE, color='blue', linestyle='dashed', marker='o',
                markerfacecolor='red', markersize=8, label='MAE')

        # Plot RMSE
        ax.plot(data_range, RMSE, color='green', linestyle='dashed', marker='s',
                markerfacecolor='yellow', markersize=8, label='RMSE')
        # Labels & Legend
        ax.set_xlabel("Number of Data Points")
        ax.set_ylabel("Error Value")
        ax.set_title("MAE & RMSE Over Data Range")
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig, use_container_width=False)
        
        st.write("The Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) rates decrease as the number of data points increase, up to around 3000 data points, where the MAE settles at around 290 seconds.")
        
        st.write("The significant margin of error is probably because the actual finish time depends on factors that are not included in our model, such as running ability and fitness level.")