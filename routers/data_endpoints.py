from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PricesRequest, QuantitiesRequest, PriceData, QuantityData
from tools.market_tools import get_commodities, get_states, get_districts, get_quantities, get_prices


router = APIRouter(prefix="/data", tags=["data"])

@router.get("/commodities")
async def get_available_commodities():
    """Get list of available commodities and states"""
    try:
        commodities = get_commodities()
        return {"commodities": commodities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching commodities: {str(e)}")

@router.get("/states")
async def get_available_states():
    """Get list of available states"""
    try:
        states = get_states()
        return {"states": states}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching states: {str(e)}")
    
@router.get("/districts/{state_id}")
async def get_districts_by_state(state_id: int):
    """Get list of districts for a given state"""
    try:
        districts = get_districts(state_id)
        return {"districts": districts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching districts for {state_id}: {str(e)}")

@router.post("/prices")
async def get_market_prices(request: PricesRequest):
    """Get market prices for a specific commodity and state"""
    try:
        # Pass the request object directly to get_prices
        price_info = get_prices(request)
        
        # For the direct API endpoint, we'll return a simplified structure
        # The full formatted response is available through the conversational AI
        return {"prices": price_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market prices: {str(e)}")


@router.post("/quantities")
async def get_market_quantities(request: QuantitiesRequest):
    """Get market quantities for a specific commodity and state"""
    try:
        # Pass the request object directly to get_quantities
        quantity_info = get_quantities(request)
        
        # For the direct API endpoint, we'll return a simplified structure
        # The full formatted response is available through the conversational AI
        return {"quantities": quantity_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market quantities: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agri-sahayak-data-api"}

