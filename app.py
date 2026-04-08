from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

# 1. Initialize the FastAPI app
app = FastAPI(
    title="EcoPredict: Global Carbon Forecast API",
    description="An AI-powered API to forecast per-capita CO2 emissions based on GDP and Energy usage."
)

@app.get("/")
async def index():
    return {"status": "Online", "message": "EcoPredict API is Live. Visit /docs for the interactive UI."}

@app.post("/predict")
async def predict_datapoint(year: int, energy: float, gdp: float, iso_code: str):
    try:
        logging.info(f"Prediction requested for {iso_code} in year {year}")
        
        # 2. Use our CustomData class to capture the input
        data = CustomData(
            year=year,
            energy_per_capita=energy,
            gdp_per_capita=gdp,
            iso_code=iso_code
        )

        # 3. Convert input into a DataFrame for the model
        pred_df = data.get_data_as_dataframe()
        logging.info("Input converted to DataFrame")

        # 4. Trigger the Prediction Pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        logging.info(f"Prediction successful: {results[0]}")

        # 5. Return JSON result
        return {
            "input": {
                "country": iso_code,
                "year": year
            },
            "forecast": {
                "co2_emissions_per_capita": float(results[0]),
                "unit": "Metric Tons"
            }
        }

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    # Start the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)