from pydantic import BaseModel, Field, confloat, constr


class FoodDescription(BaseModel):
    """Schema for food description input."""

    description: constr(min_length=1, max_length=500) = Field(
        ...,
        description="Description of the food item",
        example="2 large eggs with 1 tablespoon of olive oil",
    )


class CalorieEstimate(BaseModel):
    """Schema for individual calorie estimates."""

    value: confloat(gt=0) = Field(..., description="Estimated calories")
    confidence: confloat(ge=0, le=1) = Field(
        ..., description="Confidence score of the estimate"
    )


class CalorieEstimateResponse(BaseModel):
    """Schema for the complete calorie estimation response."""

    estimates: list[CalorieEstimate] = Field(
        ..., description="List of calorie estimates"
    )
    mean: confloat(gt=0) = Field(..., description="Mean of all estimates")
    median: confloat(gt=0) = Field(..., description="Median of all estimates")
    std_dev: confloat(ge=0) = Field(..., description="Standard deviation of estimates")
    confidence_interval: tuple[float, float] = Field(
        ..., description="95% confidence interval for the estimate"
    )
    input_description: str = Field(..., description="Original food description")
