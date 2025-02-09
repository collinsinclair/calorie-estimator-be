from openai import AsyncOpenAI
from app.core.config import settings
from datetime import datetime, timedelta
from typing import Optional
import asyncio
import numpy as np
import json

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

    async def get_single_estimate(self, food_desc: str) -> CalorieEstimate:
        """Get a single calorie estimate from OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a nutrition expert. Estimate the calories in the given food. Return a JSON object with 'calories' (a positive number) and 'confidence' (a number between 0 and 1, indicating how confident you are in the estimate).",
                    },
                    {
                        "role": "user",
                        "content": f"How many calories are in: {food_desc}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.7,  # Increased temperature for more variation
            )

            # Parse the response
            message = response.choices[0].message
            if not message.content:
                raise ValueError("No content in response")

            result = json.loads(message.content)

            # Validate and return estimate
            if "calories" not in result or "confidence" not in result:
                raise ValueError("Missing required fields in response")

            return CalorieEstimate(
                value=float(result["calories"]), confidence=float(result["confidence"])
            )

        except json.JSONDecodeError:
            raise Exception("Failed to parse JSON response from model")
        except ValueError as ve:
            raise Exception(f"Invalid response format: {str(ve)}")
        except Exception as e:
            raise Exception(f"Failed to get estimate: {str(e)}")

    async def get_multiple_estimates(
        self, food_desc: str, num_estimates: int = 10
    ) -> list[CalorieEstimate]:
        """Get multiple calorie estimates in parallel."""
        tasks = [self.get_single_estimate(food_desc) for _ in range(num_estimates)]
        return await asyncio.gather(*tasks)

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
