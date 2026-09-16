from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

_EXAMPLE_TRANSACTION = {
    "a": 4,
    "b": 0.6717,
    "c": 2201.4,
    "d": 1.0,
    "e": 0.0,
    "f": 10.0,
    "g": "AR",
    "h": 0,
    "j": "cat_a349b79",
    "k": 0.7290250494115859,
    "l": 1619.0,
    "m": 0.0,
    "n": 1,
    "o": "Y",
    "p": "N",
    "fecha": "2020-04-16T15:22:25",
    "monto": 21.33,
    "score": 9,
}


class TransactionRequest(BaseModel):
    """Raw transaction fields, exactly as expected by
    src.pipelines.preprocessing.transform (columns are anonymized in the source
    dataset; see notebooks/1_EDA.ipynb for what each one represents). Nullable
    columns here are the ones observed with missing values in the training data.
    """

    a: int
    b: Optional[float] = None
    c: Optional[float] = None
    d: Optional[float] = None
    e: float
    f: Optional[float] = None
    g: Optional[str] = None
    h: int
    j: str
    k: float = Field(..., description="Unique transaction identifier")
    l: Optional[float] = None  # noqa: E741  (dataset's own column name)
    m: Optional[float] = None
    n: int
    o: Optional[str] = None
    p: str
    fecha: datetime
    monto: float
    score: int

    model_config = {"json_schema_extra": {"example": _EXAMPLE_TRANSACTION}}


class PredictionResponse(BaseModel):
    transaction_id: float
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    threshold: float
