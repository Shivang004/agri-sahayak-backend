from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Type

class QueryRequest(BaseModel):
    """Request model for the conversational AI endpoint"""
    query: str
    language: str
    # imageUrl removed - will be handled via multipart form


class QueryResponse(BaseModel):
    """Response model for the conversational AI endpoint"""
    response: str
    agents_used: List[str]
    agent_outputs: Dict[str, str]
    mode: str
    query_en: str

class Commodity(BaseModel):
    commodity_id: int
    commodity_name: str

class Geography(BaseModel):
    census_state_id: int
    census_state_name: str
    census_district_id: int
    census_district_name: str

class State(BaseModel):
    state_id: int
    state_name: str

class District(BaseModel):
    district_id: int
    district_name: str

class PriceData(BaseModel):
    date: date
    commodity_id: int
    census_state_id: int
    census_district_id: int
    market_id: int
    min_price: float
    max_price: float
    modal_price: float

class QuantityData(BaseModel):
    date: date
    commodity_id: int
    census_state_id: int
    census_district_id: int
    market_id: int
    quantity: float

class PricesRequest(BaseModel):
    commodity_id: int
    from_date: date
    to_date: date
    state_id: Optional[int] = None
    district_id: Optional[List[int]] = None
    market_id: Optional[List[int]] = None

class QuantitiesRequest(BaseModel):
    commodity_id: int
    from_date: date
    to_date: date
    state_id: Optional[int] = None
    district_id: Optional[List[int]] = None
    market_id: Optional[List[int]] = None
    
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
