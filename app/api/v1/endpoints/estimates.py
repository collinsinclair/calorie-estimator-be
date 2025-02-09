import numpy as np
from fastapi import APIRouter

from app.models.schemas import CalorieEstimateResponse
from app.services.estimator import CalorieEstimatorService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
estimator_service = CalorieEstimatorService()


@router.post(
    "/estimate",
    response_model=CalorieEstimateResponse,
    summary="Estimate calories for a food description",
    response_description="Calorie estimation with confidence metrics",
)
async def estimate_calories(food_input: str) -> CalorieEstimateResponse:
    """
    Estimate calories for a food description using multiple LLM queries.
    """
    logger.info(f"Received estimation request: {food_input}")

    try:
        cached_response = estimator_service.get_cached_estimate(food_input)
        if cached_response:
            logger.info(f"Returning cached response for: {food_input}")
            return cached_response

        estimates = await estimator_service.get_multiple_estimates(food_input)

        if not estimates:
            logger.error("No valid estimates returned!")
            raise Exception("No valid estimates obtained.")

        analysis = estimator_service.analyze_estimates(estimates)

        confidence_interval = (
            analysis["weighted_mean"] - (1.96 * analysis["weighted_std"]),
            analysis["weighted_mean"] + (1.96 * analysis["weighted_std"]),
        )
        confidence_interval = (max(0, confidence_interval[0]), confidence_interval[1])

        response = CalorieEstimateResponse(
            estimates=estimates,
            mean=analysis["weighted_mean"],
            median=np.median([est.value for est in estimates]),
            std_dev=analysis["weighted_std"],
            confidence_interval=confidence_interval,
            input_description=food_input,
        )

        estimator_service.cache_estimate(food_input, response)

        logger.info(f"Successfully processed request: {food_input}")
        return response

    except Exception as e:
        logger.error(f"Failed to estimate calories: {e}")
        raise
