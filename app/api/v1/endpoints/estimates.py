import numpy as np
from fastapi import APIRouter

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
    Estimate calories for a food description using multiple LLM queries.
    """
    try:
        # Check cache first
        cached_response = estimator_service.get_cached_estimate(food_input.description)
        if cached_response:
            return cached_response

        # Get multiple estimates in parallel
        estimates = await estimator_service.get_multiple_estimates(
            food_input.description
        )

        # Analyze the estimates
        analysis = estimator_service.analyze_estimates(estimates)

        # Calculate confidence interval (95%)
        confidence_interval = (
            analysis["weighted_mean"] - (1.96 * analysis["weighted_std"]),
            analysis["weighted_mean"] + (1.96 * analysis["weighted_std"]),
        )

        # Ensure confidence interval lower bound is not negative
        confidence_interval = (max(0, confidence_interval[0]), confidence_interval[1])

        # Create response
        response = CalorieEstimateResponse(
            estimates=estimates,
            mean=analysis["weighted_mean"],
            median=np.median([est.value for est in estimates]),
            std_dev=analysis["weighted_std"],
            confidence_interval=confidence_interval,
            input_description=food_input.description,
        )

        # Cache the response
        estimator_service.cache_estimate(food_input.description, response)

        return response

    except Exception as e:
        raise Exception(f"Failed to estimate calories: {str(e)}")
