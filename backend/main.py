import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pipeline.engine import OptimizedExoplanetPipeline

app = FastAPI(
    title="Kepler Exoplanet Detection API",
    description="Backend service powering transit signal extraction and evaluation[cite: 1].",
    version="1.0.0"
)

# Enable CORS for local frontend development (e.g., React on port 3000 / 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = OptimizedExoplanetPipeline(sde_threshold=10.0)

@app.get("/api/health")
def health_check():
    """Returns engine readiness status[cite: 1]."""
    return {
        "status": "online",
        "engine": "OptimizedExoplanetPipeline",
        "sde_threshold": pipeline.sde_threshold
    }

@app.post("/api/analyze")
async def analyze_star(
    star_id: str = Query(..., description="Target Star Identifier (e.g. STAR_0043)[cite: 1]"),
    file: UploadFile = File(...)
):
    """Processes an uploaded Kepler Parquet light curve file[cite: 1]."""
    if not file.filename.endswith(('.parquet', '.pq')):
        raise HTTPException(status_code=400, detail="Invalid file format. Upload a valid Parquet file[cite: 1].")

    try:
        contents = await file.read()
        df = pd.read_parquet(io.BytesIO(contents))
        
        # Verify required schema columns[cite: 1]
        required_cols = {'time', 'flux', 'flux_err', 'quality', 'quarter'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Parquet missing required columns: {missing}[cite: 1]")

        result = pipeline.process_star(star_id, df)
        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")