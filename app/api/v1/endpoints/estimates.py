from fastapi import APIRouter, HTTPException
from app.models.schemas import FoodDescription, CalorieEstimateResponse
from app.services.estimator import CalorieEstimatorService

router = APIRouter()
estimator_service = CalorieEstimatorService()


@router.post(
    "/estimate",
    response_model=CalorieEstimateResponse,
    summary="Estimate calories for a food description",
    response_description="Calorie estimation with confidence metrics",
)
async def estimate_calories(food_input: FoodDescription) -> CalorieEstimateResponse:
    """
    Estimate calories for a given food description.

    - Uses multiple API calls to generate estimates
    - Provides statistical analysis of the estimates
    - Returns standardized food description

    Args:
        food_input: Food description with optional unit system

    Returns:
        Calorie estimation with confidence metrics and statistical analysis

    Raises:
        HTTPException: If the estimation fails
    """
    try:
        return await estimator_service.estimate_calories(food_input)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to estimate calories: {str(e)}"
        )
