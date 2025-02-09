import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

from app.core.config import settings
from app.models.schemas import CalorieEstimateResponse, CalorieEstimate
from app.models.schemas import CalorieReasoning


class CacheEntry:
    def __init__(self, value: CalorieEstimateResponse):
        self.value = value
        self.timestamp = datetime.now()


class CalorieEstimatorService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.cache = {}
        self.cache_ttl = timedelta(hours=24)
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    async def get_single_estimate(self, food_desc: str) -> CalorieEstimate:
        """Get a single calorie estimate from OpenAI with structured reasoning."""
        self.logger.debug(f"Requesting calorie estimate for: {food_desc}")

        try:
            completion = await self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a nutrition expert. Break down food items into their components
                        and calculate calories step by step. For each step:
                        1. Explain your analysis
                        2. List the specific components and their calorie contributions
                        3. Keep a running subtotal
                        Finally, provide your total estimate and confidence level based on your analysis.""",
                    },
                    {
                        "role": "user",
                        "content": f"Calculate the calories in: {food_desc}",
                    },
                ],
                response_format=CalorieReasoning,
            )

            message = completion.choices[0].message

            if message.refusal:
                self.logger.warning(f"Model refused to analyze: {food_desc}")
                self.logger.warning(f"Refusal message: {message.refusal}")
                raise ValueError(f"Model refused analysis: {message.refusal}")

            reasoning = message.parsed

            # Log the reasoning steps
            for step in reasoning.steps:
                self.logger.debug(f"Reasoning step: {step.explanation}")
                for component in step.components:
                    self.logger.debug(
                        f"Component: {component.name} = {component.calories} calories"
                        f" ({component.explanation})"
                    )
                self.logger.debug(f"Subtotal after step: {step.subtotal}")

            # Create CalorieEstimate from the reasoned result
            estimate = CalorieEstimate(
                value=reasoning.final_estimate, confidence=reasoning.confidence
            )

            return estimate

        except Exception as e:
            self.logger.error(f"Failed to get estimate for {food_desc}: {e}")
            raise

    async def get_multiple_estimates(
        self, food_desc: str, num_estimates: int = 5
    ) -> list[CalorieEstimate]:
        """Get multiple calorie estimates in parallel."""
        self.logger.debug(
            f"Starting {num_estimates} parallel requests for: {food_desc}"
        )

        tasks = [self.get_single_estimate(food_desc) for _ in range(num_estimates)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log errors if any task failed
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Task {i} failed: {result}")

        self.logger.debug(f"Completed parallel requests for: {food_desc}")

        # Filter out failed requests
        return [r for r in results if isinstance(r, CalorieEstimate)]

    def analyze_estimates(self, estimates: list[CalorieEstimate]) -> dict:
        """Perform statistical analysis on estimates."""
        values = [est.value for est in estimates]
        confidences = [est.confidence for est in estimates]

        # Weight estimates by confidence
        weighted_mean = np.average(values, weights=confidences)

        # Calculate weighted standard deviation
        weighted_var = np.average((values - weighted_mean) ** 2, weights=confidences)
        weighted_std = np.sqrt(weighted_var)

        return {"weighted_mean": weighted_mean, "weighted_std": weighted_std}

    def get_cached_estimate(
        self, description: str
    ) -> Optional[CalorieEstimateResponse]:
        """Get cached estimate if available and not expired."""
        if description in self.cache:
            entry = self.cache[description]
            if datetime.now() - entry.timestamp < self.cache_ttl:
                return entry.value
            else:
                del self.cache[description]
        return None

    def cache_estimate(self, description: str, response: CalorieEstimateResponse):
        """Cache estimation response."""
        self.cache[description] = CacheEntry(response)

        # Clean old entries
        current_time = datetime.now()
        expired_keys = [
            k
            for k, v in self.cache.items()
            if current_time - v.timestamp > self.cache_ttl
        ]
        for k in expired_keys:
            del self.cache[k]
