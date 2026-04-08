import matplotlib
# MUST be at the very top to prevent the popup window
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import shap
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import os

# 1. Create fake user data
# Test Case: China (CHN)
data = CustomData(
    year=2023,
    energy_per_capita=1500.707,
    gdp_per_capita=2000.136473023602,
    iso_code="USA"
)

# 2. Convert to DF (Ensure method name matches your CustomData class)
final_df = data.get_data_as_dataframe()
print("Input Data:")
print(final_df)

# 3. Predict
predict_pipeline = PredictPipeline()
result = predict_pipeline.predict(final_df)
print(f"\nPredicted CO2: {result[0]}")

# 4. Explain with SHAP
print("\nGenerating SHAP explanation...")
shap_values, data_transformed, feature_names = predict_pipeline.explain(final_df)

# 5. Correct Plotting Logic
plt.clf() # Clear current figure
fig = plt.figure(figsize=(12, 8))

# CRUCIAL: show=False keeps the plot open so we can save it
shap.summary_plot(
    shap_values, 
    data_transformed, 
    feature_names=feature_names, 
    plot_type="bar", 
    show=False
)

# Save with 'tight' layout to ensure labels aren't cut off
plt.title(f"CO2 Prediction Explanation for {data.iso_code}")
plt.savefig("artifacts/explanations234.png", bbox_inches='tight', dpi=300)
plt.close(fig)

print("✅ SHAP explanation saved as artifacts/shap_explanation.png")