from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# 1. Create fake user data
data = CustomData(
    year=2023,
    energy_per_capita=3123555.707,
    gdp_per_capita=189.136473023602,
    iso_code="CHN"
)

# 2. Convert to DF
final_df = data.get_data_as_dataframe()
print(final_df)

# 3. Predict
predict_pipeline = PredictPipeline()
result = predict_pipeline.predict(final_df)

print(f"Predicted CO2: {result[0]}")