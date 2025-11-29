# Disease_Detection/disease_main.py
#
# Unified disease detection service for rice + wheat
# This runs in the TensorFlow environment (tf_env).
#
# Exposes:
#   GET  /health
#   (rice_Api.py endpoints) mounted at  /rice
#   (wheat_Api.py endpoints) mounted at /wheat

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your existing apps
from Disease_Detection.rice_Api import app as rice_app
from Disease_Detection.wheat_Api import app as wheat_app

app = FastAPI(
    title="AgroAI Disease Detection Service",
    description=(
        "TensorFlow-based disease detection for rice and wheat.\n"
        "Rice API mounted at /rice\n"
        "Wheat API mounted at /wheat"
    ),
    version="1.0.0",
)

# CORS – keep it wide open for now; tighten later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount rice and wheat sub-apps
app.mount("/rice", rice_app)
app.mount("/wheat", wheat_app)


@app.get("/health")
async def health():
    """Basic health check for the disease service."""
    return {
        "status": "ok",
        "service": "disease",
        "rice_mounted": True,
        "wheat_mounted": True,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "Disease_Detection.disease_main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
