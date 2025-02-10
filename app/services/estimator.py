import asyncio
import logging

import numpy as np
from openai import AsyncOpenAI

from app.core.config import settings
from app.models.schemas import CalorieEstimate, FoodDescription
from app.models.schemas import CalorieReasoning


class CalorieEstimatorService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        # Log API key presence (not the key itself)
        self.logger.debug(f"OpenAI API Key present: {bool(settings.OPENAI_API_KEY)}")

    async def get_single_estimate(self, food_desc: FoodDescription) -> CalorieEstimate:
        """Get a single calorie estimate from OpenAI with structured reasoning."""
        self.logger.debug(
            f"Starting single estimate request for: {food_desc.description}"
        )

        try:
            self.logger.debug("Initiating OpenAI API call...")
            completion = await self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """
                    Estimate the caloric content of the food/dish described. In your analysis, please follow these steps:

                    1. **Identify Explicit Ingredients:** List the ingredients that are directly mentioned in the description.
                    2. **Infer Implied Ingredients or Processes:** Consider common or expected ingredients and preparation processes that might not be explicitly stated. For example, if the input is "CALIFORNIA BURGER: melted Monterey jack cheese, avocado, arugula and red onion", infer that a beef patty and a bun are likely included.
                    3. **Estimate Caloric Contributions:** For each explicit and inferred ingredient, estimate its typical caloric contribution.
                    4. **Provide Reasoning Steps:** Include subtotals for each reasoning step, ensuring that the cumulative reasoning is clear and consistent.
                    5. **Sanity-Check Your Estimate:** Validate your final estimate by comparing it against typical portion sizes or expected weight ranges. If the result seems off, re-evaluate your reasoning The step explanation for this step should start with "SANITY CHECK:".
                    6. **Return Structured Output:** Output your final estimated calorie count along with the complete breakdown of your reasoning process.

                    Follow the above steps to produce a detailed, step-by-step analysis that accounts for both explicit and implicit aspects of the food description.
                    """,
                    },
                    {
                        "role": "user",
                        "content": f"Estimate the caloric content of {food_desc.description}",
                    },
                ],
                response_format=CalorieReasoning,
                store=True,
            )
            self.logger.debug("OpenAI API call completed successfully")

            message = completion.choices[0].message
            self.logger.debug(f"Message received: {message}")

            if hasattr(message, "refusal") and message.refusal:
                self.logger.warning(
                    f"Model refused to analyze: {food_desc.description}"
                )
                self.logger.warning(f"Refusal message: {message.refusal}")
                raise ValueError(f"Model refused analysis: {message.refusal}")

            reasoning = message.parsed
            self.logger.debug(f"Parsed reasoning: {reasoning}")

            # Log the reasoning steps
            for i, step in enumerate(reasoning.steps):
                self.logger.debug(f"Reasoning step {i + 1}: {step.explanation}")
                for component in step.components:
                    self.logger.debug(
                        f"Component: {component.name} = {component.calories} calories"
                        f" ({component.explanation})"
                    )
                self.logger.debug(f"Subtotal after step {i + 1}: {step.subtotal}")

            # Create CalorieEstimate from the reasoned result
            estimate = CalorieEstimate(value=reasoning.final_estimate)
            self.logger.debug(f"Created estimate: {estimate}")

            return estimate

        except Exception as e:
            self.logger.error(
                f"Failed to get estimate for {food_desc.description}: {str(e)}",
                exc_info=True,
            )
            raise

    async def get_multiple_estimates(
        self, food_desc: FoodDescription, num_estimates: int = 100
    ) -> list[CalorieEstimate]:
        """Get multiple calorie estimates in parallel."""
        self.logger.debug(
            f"Starting {num_estimates} parallel requests for: {food_desc.description}"
        )

        tasks = [self.get_single_estimate(food_desc) for _ in range(num_estimates)]

        self.logger.debug("Gathering parallel tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log details about each task result
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Task {i} failed with error: {str(result)}", exc_info=True
                )
            else:
                self.logger.debug(f"Task {i} succeeded with estimate: {result}")

        self.logger.debug(f"Completed parallel requests for: {food_desc.description}")

        # Filter out failed requests
        valid_results = [r for r in results if isinstance(r, CalorieEstimate)]
        self.logger.debug(
            f"Number of valid results: {len(valid_results)} out of {num_estimates}"
        )

        return valid_results

    def analyze_estimates(self, estimates: list[CalorieEstimate]) -> dict:
        """Perform statistical analysis on estimates."""
        self.logger.debug(f"Analyzing {len(estimates)} estimates")

        values = [est.value for est in estimates]
        self.logger.debug(f"Values: {values}")

        mean = np.mean(values)
        std = np.std(values)

        self.logger.debug(f"Calculated mean: {mean}")
        self.logger.debug(f"Calculated std dev: {std}")

        return {"mean": mean, "std": std}
