from openai import OpenAI
from app.core.config import settings
from datetime import datetime, timedelta
from typing import Optional
import asyncio
import re
import numpy as np

from app.models.schemas import CalorieEstimateResponse, CalorieEstimate, UnitSystem


class CacheEntry:
    def __init__(self, value: CalorieEstimateResponse):
        self.value = value
        self.timestamp = datetime.now()


class CalorieEstimatorService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.cache = {}
        self.cache_ttl = timedelta(hours=24)

    async def get_single_estimate(self, standardized_desc: str) -> CalorieEstimate:
        """Get a single calorie estimate from OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_schema"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a nutrition expert. Estimate calories for food descriptions accurately.",
                    },
                    {
                        "role": "user",
                        "content": f"Estimate calories for: {standardized_desc}",
                    },
                ],
                temperature=0.2,
            )

            # Parse response and extract estimate
            result = response.choices[0].message.content
            # Add parsing logic here

            return CalorieEstimate(
                value=float(result["calories"]), confidence=float(result["confidence"])
            )
        except Exception as e:
            raise Exception(f"Failed to get estimate: {str(e)}")

    async def get_multiple_estimates(
        self, standardized_desc: str, num_estimates: int = 3
    ) -> list[CalorieEstimate]:
        """Get multiple calorie estimates in parallel."""
        tasks = [
            self.get_single_estimate(standardized_desc) for _ in range(num_estimates)
        ]
        return await asyncio.gather(*tasks)

    def convert_to_metric(self, amount: float, from_unit: str) -> tuple[float, str]:
        """Convert measurements to metric system."""
        conversions = {
            "cup": 236.588,  # ml
            "tablespoon": 14.787,  # ml
            "teaspoon": 4.929,  # ml
            "pound": 453.592,  # g
            "ounce": 28.3495,  # g
            "inch": 2.54,  # cm
        }

        if from_unit in conversions:
            return amount * conversions[from_unit], (
                "ml" if from_unit in ["cup", "tablespoon", "teaspoon"] else "g"
            )
        return amount, from_unit

    def standardize_description(self, description: str, unit_system: UnitSystem) -> str:
        """
        Standardize food description with unit conversion.
        """
        # Common unit patterns
        unit_patterns = {
            r"(\d+(?:\.\d+)?)\s*(tbsp|tbs|tablespoons?)": "tablespoon",
            r"(\d+(?:\.\d+)?)\s*(tsp|teaspoons?)": "teaspoon",
            r"(\d+(?:\.\d+)?)\s*(cups?)": "cup",
            r"(\d+(?:\.\d+)?)\s*(oz|ounces?)": "ounce",
            r"(\d+(?:\.\d+)?)\s*(lbs?|pounds?)": "pound",
        }

        standardized = description.lower()

        for pattern, unit in unit_patterns.items():
            matches = re.finditer(pattern, standardized)
            for match in matches:
                amount = float(match.group(1))
                if unit_system == UnitSystem.METRIC:
                    converted_amount, converted_unit = self.convert_to_metric(
                        amount, unit
                    )
                    standardized = standardized.replace(
                        match.group(0), f"{converted_amount:.1f} {converted_unit}"
                    )

        return standardized

    def analyze_estimates(self, estimates: list[CalorieEstimate]) -> dict:
        """
        Perform detailed statistical analysis on estimates.
        """
        values = [est.value for est in estimates]
        confidences = [est.confidence for est in estimates]

        # Weight estimates by confidence
        weighted_mean = np.average(values, weights=confidences)

        # Calculate weighted standard deviation
        weighted_var = np.average((values - weighted_mean) ** 2, weights=confidences)
        weighted_std = np.sqrt(weighted_var)

        # Detect outliers using IQR method
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        filtered_values = [v for v in values if lower_bound <= v <= upper_bound]

        return {
            "weighted_mean": weighted_mean,
            "weighted_std": weighted_std,
            "filtered_mean": (
                np.mean(filtered_values) if filtered_values else weighted_mean
            ),
            "outliers_removed": len(values) - len(filtered_values),
        }

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
