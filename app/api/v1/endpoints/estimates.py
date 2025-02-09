import numpy as np
from fastapi import APIRouter

from app.models.schemas import CalorieEstimateResponse, FoodDescription
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
async def estimate_calories(food_input: FoodDescription) -> CalorieEstimateResponse:
    """
    Estimate calories for a food description using multiple LLM queries.
    """
    logger.info(f"Received estimation request for: {food_input.description}")

    try:
        logger.debug("Calling estimator service for multiple estimates...")
        estimates = await estimator_service.get_multiple_estimates(food_input)
        logger.debug(f"Received {len(estimates)} estimates from service")

        if not estimates:
            logger.error("No valid estimates returned from service!")
            logger.error("Food description that failed: {food_input.description}")
            raise Exception("No valid estimates obtained.")

        logger.debug("Analyzing estimates...")
        analysis = estimator_service.analyze_estimates(estimates)
        logger.debug(f"Analysis results: {analysis}")

        confidence_interval = (
            analysis["weighted_mean"] - (1.96 * analysis["weighted_std"]),
            analysis["weighted_mean"] + (1.96 * analysis["weighted_std"]),
        )
        confidence_interval = (
            max(0, round(confidence_interval[0])),
            round(confidence_interval[1]),
        )
        logger.debug(f"Calculated confidence interval: {confidence_interval}")

        response = CalorieEstimateResponse(
            estimates=estimates,
            mean=round(analysis["weighted_mean"]),
            median=round(np.median([est.value for est in estimates])),
            std_dev=round(analysis["weighted_std"]),
            confidence_interval=confidence_interval,
            input_description=food_input.description,
        )

        logger.info(f"Successfully processed request. Response: {response}")
        return response

    except Exception as e:
        logger.error(f"Failed to estimate calories: {str(e)}", exc_info=True)
        raise
