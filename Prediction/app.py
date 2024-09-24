import pandas as pd
import numpy as np
import pickle
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
#Load the Trained model 
with open(r'Prediction/Best_Random_Forest_model.pkl','rb') as f:
    model=pickle.load(f)

# Load the dataset for visualization
path=r'Data/synthetic_COPD_data.csv'
df=pd.read_csv(path)

# Streamlit App
def prediction_dashboard():
    st.title('COPD Prediction Analysis Dashboard')
    
    # User Input
    st.sidebar.header('Data Filter Option')
    age = st.sidebar.slider("Age", 30, 88, 50)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    bmi = st.sidebar.slider("BMI", 10, 40, 25)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["Current", "Former", "Never"])
    biomass_fuel_exposure = st.sidebar.selectbox("Biomass Fuel Exposure", ["Yes", "No"])
    occupational_exposure = st.sidebar.selectbox("Occupational Exposure", ["Yes", "No"])
    family_history = st.sidebar.selectbox("Family History", ["Yes", "No"])
    air_pollution_level = st.sidebar.slider("Air Pollution Level", 0, 300, 50)
    respiratory_infections = st.sidebar.selectbox("Respiratory Infections in Childhood", ["Yes", "No"])
    location = st.sidebar.selectbox("Location", ['Chitwan', 'Dharan', 'Hetauda', 'Kathmandu', 'Lalitpur', 'Nepalgunj', 'Pokhara'])

    # Process the input data
    input_data = {
        'Age': [age],
        'Gender': [gender],
        'Biomass_Fuel_Exposure': [biomass_fuel_exposure],
        'Occupational_Exposure': [occupational_exposure],
        'Family_History_COPD': [family_history],
        'BMI': [bmi],
        'Air_Pollution_Level': [air_pollution_level],
        'Respiratory_Infections_Childhood': [respiratory_infections],
        'Smoking_Status_encoded': [smoking_status],
        'Location': [location]
    }

    # Convert the data to a DataFrame
    input_df = pd.DataFrame(input_data)

    # Encode categorical features
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    input_df['Smoking_Status_encoded'] = input_df['Smoking_Status_encoded'].map({'Current': 1, 'Former': 0.5, 'Never': 0})
    input_df['Biomass_Fuel_Exposure'] = input_df['Biomass_Fuel_Exposure'].map({'Yes': 1, 'No': 0})
    input_df['Occupational_Exposure'] = input_df['Occupational_Exposure'].map({'Yes': 1, 'No': 0})
    input_df['Family_History_COPD'] = input_df['Family_History_COPD'].map({'Yes': 1, 'No': 0})
    input_df['Respiratory_Infections_Childhood'] = input_df['Respiratory_Infections_Childhood'].map({'Yes': 1, 'No': 0})

    # One-Hot Encoding for the Location variable
    location_one_hot = pd.get_dummies(input_df['Location'], drop_first=True)

    # Drop the original 'Location' column and append the one-hot encoded columns
    input_df = pd.concat([input_df.drop(columns=['Location']), location_one_hot], axis=1)

    # Ensure that all necessary columns are present, even if they are zero-filled
    location_columns = ['Location_Biratnagar', 'Location_Butwal', 'Location_Chitwan', 
                        'Location_Dharan', 'Location_Hetauda', 'Location_Kathmandu', 
                        'Location_Lalitpur', 'Location_Nepalgunj', 'Location_Pokhara']
    
    for col in location_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing location columns with 0

    # Prediction and Analysis
    if st.button("Predict"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Display the prediction result
        if prediction[0] == 1:
            st.write("Prediction: **COPD Detected**")
        else:
            st.write("Prediction: **No COPD Detected**")
        
        # Show the prediction probability
        st.write(f"Prediction Probability: COPD: {prediction_proba[0][1]:.2f}, No COPD: {prediction_proba[0][0]:.2f}")

        # Additional Analysis
        st.subheader("Analysis Based on Input Data")
        
        # Compare age and BMI to dataset averages
        avg_age = df['Age'].mean()
        avg_bmi = df['BMI'].mean()

        st.write(f"Your Age: **{age}** (Average Age in Dataset: {avg_age:.1f})")
        st.write(f"Your BMI: **{bmi}** (Average BMI in Dataset: {avg_bmi:.1f})")

        # Check if high-risk factors are present
        risk_factors = []
        if smoking_status == "Current":
            risk_factors.append("Smoking Status: **Current Smoker**")
        if family_history == "Yes":
            risk_factors.append("Family History of COPD: **Yes**")
        if biomass_fuel_exposure == "Yes":
            risk_factors.append("Biomass Fuel Exposure: **Yes**")
        if occupational_exposure == "Yes":
            risk_factors.append("Occupational Exposure: **Yes**")
        if air_pollution_level > df['Air_Pollution_Level'].mean():
            risk_factors.append(f"Air Pollution Level: **{air_pollution_level}** (Above average pollution)")

        if risk_factors:
            st.write("### Identified High-Risk Factors:")
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.write("No major high-risk factors identified based on input data.")

        # Visualize input features against dataset statistics
        st.subheader("Comparison to Population")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Age Comparison
        sns.histplot(df['Age'], kde=True, color='blue', ax=ax[0])
        ax[0].axvline(age, color='red', linestyle='--', label=f"Your Age: {age}")
        ax[0].set_title('Age Distribution')
        ax[0].legend()

        # Plot BMI Comparison
        sns.histplot(df['BMI'], kde=True, color='green', ax=ax[1])
        ax[1].axvline(bmi, color='red', linestyle='--', label=f"Your BMI: {bmi}")
        ax[1].set_title('BMI Distribution')
        ax[1].legend()

        st.pyplot(fig)

def visualization_dashboard():
    st.title("COPD Data Exploration")

    # Sidebar filter options
    st.sidebar.header("Filter Data")
    
    # Filter by Age
    age_range = st.sidebar.slider("Select Age Range", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=(30, 88))
    
    # Filter by Gender
    gender_filter = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    
    # Filter by Smoking Status
    smoking_filter = st.sidebar.multiselect("Select Smoking Status", options=df['Smoking_Status'].unique(), default=df['Smoking_Status'].unique())
    
    # Filter by BMI Range
    bmi_range = st.sidebar.slider("Select BMI Range", min_value=float(df['BMI'].min()), max_value=float(df['BMI'].max()), value=(10.0, 40.0))
    
    # Filter by Location
    location_filter = st.sidebar.multiselect("Select Location", options=df['Location'].unique(), default=df['Location'].unique())
    
    # Submit button for filtering
    if st.sidebar.button("Submit"):
        # Apply the filters to the dataset
        filtered_df = df[
            (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
            (df['Gender'].isin(gender_filter)) &
            (df['Smoking_Status'].isin(smoking_filter)) &
            (df['BMI'] >= bmi_range[0]) & (df['BMI'] <= bmi_range[1]) &
            (df['Location'].isin(location_filter))
        ]

        st.subheader("Filtered Data Overview")
        st.dataframe(filtered_df)  # Display the filtered data

        # Univariate Analysis for Filtered Data
        st.subheader("Age Distribution (Filtered Data)")
        plt.figure(figsize=(14, 8))
        sns.histplot(filtered_df['Age'], kde=True, color='blue')
        plt.title('Age Distribution (Filtered Data)')
        st.pyplot(plt)
        plt.clf()

        st.subheader("BMI Distribution (Filtered Data)")
        plt.figure(figsize=(14, 8))
        sns.histplot(filtered_df['BMI'], kde=True, color='green')
        plt.title('BMI Distribution (Filtered Data)')
        st.pyplot(plt)
        plt.clf()

        st.subheader("Gender Distribution (Filtered Data)")
        plt.figure(figsize=(14, 8))
        sns.countplot(x='Gender', data=filtered_df, palette='viridis')
        plt.title('Gender Distribution (Filtered Data)')
        st.pyplot(plt)
        plt.clf()

        st.subheader("Smoking Status Distribution (Filtered Data)")
        plt.figure(figsize=(14, 8))
        sns.countplot(x='Smoking_Status', data=filtered_df, palette='Set1')
        plt.title('Smoking Status Distribution (Filtered Data)')
        st.pyplot(plt)
        plt.clf()

        # Bivariate Analysis for Filtered Data
        st.subheader("Age vs COPD Diagnosis (Filtered Data)")
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='COPD_Diagnosis', y='Age', data=filtered_df, palette='coolwarm')
        plt.title('Age vs COPD Diagnosis (Filtered Data)')
        st.pyplot(plt)
        plt.clf()

        st.subheader("BMI vs COPD Diagnosis (Filtered Data)")
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='COPD_Diagnosis', y='BMI', data=filtered_df, palette='coolwarm')
        plt.title('BMI vs COPD Diagnosis (Filtered Data)')
        st.pyplot(plt)
        plt.clf()

        st.subheader("Smoking Status vs COPD Diagnosis (Filtered Data)")
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Smoking_Status', hue='COPD_Diagnosis', data=filtered_df, palette='Set2')
        plt.title('Smoking Status vs COPD Diagnosis (Filtered Data)')
        st.pyplot(plt)
        plt.clf()

        st.subheader("Gender vs COPD Diagnosis Count (Filtered Data)")
        plt.figure(figsize=(14, 8))
        sns.countplot(x='Gender', hue='COPD_Diagnosis', data=filtered_df, palette='Set2')
        plt.title("Gender vs COPD Diagnosis Count (Filtered Data)")
        st.pyplot(plt)
        plt.clf()

        st.subheader("Correlation Matrix (Filtered Data)")
        data_corr_filtered = filtered_df[['Age', 'Biomass_Fuel_Exposure', 'Occupational_Exposure', 'Family_History_COPD', 'BMI', 'Air_Pollution_Level', 'Respiratory_Infections_Childhood', 'COPD_Diagnosis']]
        corr_filtered = data_corr_filtered.corr()
        plt.figure(figsize=(14, 8))
        sns.heatmap(corr_filtered, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix (Filtered Data)')
        st.pyplot(plt)
        plt.clf()

# Main app function
def main():
    st.sidebar.title("App Menu")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Data Visualization", "COPD Prediction"])

    if app_mode == "COPD Prediction":
        prediction_dashboard()
    elif app_mode == "Data Visualization":
        visualization_dashboard()


# Main app function
def main():
    st.sidebar.title("App Menu")
    app_mode = st.sidebar.selectbox("Choose the app mode",["Data Visualization","COPD Prediction"])

    if app_mode == "COPD Prediction":
        prediction_dashboard()
    elif app_mode == "Data Visualization":
        visualization_dashboard()
        
if __name__ == "__main__":
    main()
