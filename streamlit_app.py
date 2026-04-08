import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

# 1. Page Configuration
st.set_page_config(
    page_title="EcoPredict: Global Carbon Forecast",
    page_icon="🌍",
    layout="wide"
)

# 2. Styling
st.title("🌍 EcoPredict: AI Carbon Intelligence")
st.markdown("""
    This system forecasts **CO2 Emissions per capita** based on economic and energy indicators. 
    Using an **XGBoost Regressor** trained on 20 years of global data.
""")

st.sidebar.header("User Input Parameters")

# 3. Input Form in Sidebar
def user_input_features():
    # We use realistic ranges based on our raw_data.csv
    iso_code = st.sidebar.selectbox("Select Country Code", 
                                   options=["USA", "IND", "CHN", "NOR", "ALB", "AND", "GBR", "FRA", "CAN"])
    year = st.sidebar.slider("Forecast Year", 2023, 2030, 2024)
    energy = st.sidebar.number_input("Energy Use per Capita (kWh)", min_value=0.0, value=2500.0)
    gdp = st.sidebar.number_input("GDP per Capita (Current $)", min_value=0.0, value=5000.0)
    
    return CustomData(
        year=year,
        energy_per_capita=energy,
        gdp_per_capita=gdp,
        iso_code=iso_code
    )

custom_data = user_input_features()

# 4. Main Prediction Logic
if st.sidebar.button("Generate Forecast"):
    try:
        # Convert input to DataFrame
        input_df = custom_data.get_data_as_dataframe()
        
        # Display inputs
        st.subheader("Current Input Summary")
        st.write(input_df)

        # Initialize Pipeline
        pipeline = PredictPipeline()
        
        # Perform Prediction
        with st.spinner("Calculating environmental impact..."):
            prediction = pipeline.predict(input_df)
            st.success(f"###### Predicted CO2: {prediction[0]:.4f} Metric Tons per capita")

        # 5. SHAP Explainability Section
        st.divider()
        st.subheader("🔍 Explainable AI: Why this prediction?")
        st.write("The chart below shows how each feature contributed to the final forecast.")

        with st.spinner("Generating SHAP explanation..."):
            shap_values, data_transformed, feature_names = pipeline.explain(input_df)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(4, 2))
            shap.summary_plot(
                shap_values, 
                data_transformed, 
                feature_names=feature_names, 
                plot_type="bar", 
                show=False
            )
            st.pyplot(plt.gcf())
            
        st.info("""
            **How to read this:** 
            - **Longer bars** mean the feature had a bigger impact.
            - `energy_per_capita` is usually the strongest driver.
            - The `iso_code` bars show the 'Carbon Signature' of the specific country.
        """)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logging.error(f"Streamlit Error: {e}")

else:
    st.info("👈 Enter country data in the sidebar and click **Generate Forecast** to begin.")

st.sidebar.markdown("---")
st.sidebar.subheader("Admin & Monitoring")

if st.sidebar.button("Run Health Audit"):
    try:
        from src.pipeline.monitoring_pipeline import ModelMonitoring
        monitor = ModelMonitoring()
        
        # In a real app, you'd pull from your 'prediction_logs' table
        # For now, we compare the model against a sample of its own training data
        sample_data = pd.read_csv("data/raw_data/raw_data.csv").sample(100)
        
        with st.spinner("Generating Statistical Audit..."):
            report_path = monitor.initiate_monitoring(sample_data)
            
        st.sidebar.success("Audit Complete!")
        
        # Create a button to open the HTML report
        with open(report_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
            st.components.v1.html(html_data, height=1000, scrolling=True)
            
    except Exception as e:
        st.sidebar.error(f"Audit failed: {e}")


# 6. MLOps Footer
st.sidebar.markdown("---")
st.sidebar.caption("System Status: 🟢 Stable")
st.sidebar.caption("Model Version: XGBoost-v2.1 (Clean Data)")