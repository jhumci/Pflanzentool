from pydantic import BaseModel
from typing import Dict, List, Optional

class NutrientProfile(BaseModel):
    n: float = 0.0  # Stickstoff
    p: float = 0.0  # Phosphor (elementar)
    k: float = 0.0  # Kalium (elementar)
    ca: float = 0.0 # Calcium
    mg: float = 0.0 # Magnesium
    s: float = 0.0  # Schwefel

class GrowthPhase(BaseModel):
    name: str
    target: NutrientProfile

class Fertilizer(BaseModel):
    name: str
    composition: NutrientProfile  # mg pro ml oder mg pro g
    price_per_ml: float
    is_liquid: bool = True

class Plant(BaseModel):
    name: str
    phases: Dict[str, NutrientProfile]