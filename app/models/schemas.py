from pydantic import BaseModel, Field, confloat


class CalorieEstimate(BaseModel):
    """Schema for individual calorie estimates."""

    value: confloat(gt=0) = Field(..., description="Estimated calories")


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


class FoodDescription(BaseModel):
    """Schema for a food description."""

    description: str = Field(..., description="Food description")


class CalorieComponent(BaseModel):
    """Schema for individual food component calorie analysis."""

    name: str = Field(..., description="Name of the food component")
    calories: float = Field(..., description="Calories in this component")
    explanation: str = Field(..., description="Explanation of the calorie calculation")


class ReasoningStep(BaseModel):
    """Schema for a single step in calorie estimation reasoning."""

    explanation: str = Field(..., description="Explanation of this reasoning step")
    components: list[CalorieComponent] = Field(
        ..., description="Calorie components identified in this step"
    )
    subtotal: float = Field(..., description="Running calorie subtotal after this step")


class CalorieReasoning(BaseModel):
    """Schema for the complete calorie estimation reasoning."""

    steps: list[ReasoningStep] = Field(..., description="Steps in estimation process")
    final_estimate: float = Field(..., description="Final calorie estimate")
