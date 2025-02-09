from openai import AsyncOpenAI
from app.core.config import settings
from datetime import datetime, timedelta
from typing import Optional
import asyncio
import numpy as np
import json
import logging


from app.models.schemas import CalorieEstimateResponse, CalorieEstimate


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
        """Get a single calorie estimate from OpenAI."""
        self.logger.debug(f"Requesting calorie estimate for: {food_desc}")
        try:
            response = await self.client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a nutrition expert. Estimate the calories in the given food. First think through the estimate, then return a JSON object with 'calories' (a positive number representing your final estimate) and 'confidence' (a number between 0 and 1, indicating how confident you are in the estimate). Use minimal whitespace in your response.",
                    },
                    {
                        "role": "user",
                        "content": f"How many calories are in: {food_desc}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.8,
                store=True
            )

            self.logger.debug(f"Received response for {food_desc}: {response}")

            # Parse and validate response
            message = response.choices[0].message
            if not message.content:
                raise ValueError("No content in response")

            result = json.loads(message.content)

            if "calories" not in result or "confidence" not in result:
                raise ValueError("Missing required fields in response")

            estimate = CalorieEstimate(
                value=float(result["calories"]), confidence=float(result["confidence"])
            )
            self.logger.debug(f"Parsed estimate: {estimate}")
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
