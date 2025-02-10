from fastapi import APIRouter

from app.api.v1.endpoints import estimates

api_router = APIRouter()
api_router.include_router(estimates.router, prefix="/estimates", tags=["estimates"])
