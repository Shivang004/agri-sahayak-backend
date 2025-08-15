import os
import requests
import json
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Type
from models import Commodity, Geography, State, District, PriceData, QuantityData, PricesRequest, QuantitiesRequest
from pydantic import BaseModel, TypeAdapter, ValidationError

# --- Configuration ---
# The base URL for the CEDA API v1
BASE_URL = os.getenv("CEDA_API_URL", "https://api.ceda.ashoka.edu.in/v1")

# Retrieve the API token from an environment variable.
# A default test key is provided, but it's best to use your own.
API_KEY = os.getenv("CEDA_API_KEY", "920c929e3b63d8febc3c041c0cd9984ce71e7a107d019be6425c2d0e4b24fd6c")

if not API_KEY:
    raise ValueError("CEDA_API_KEY environment variable not set. Please get a token from https://api.ceda.ashoka.edu.in.")

# Set up a requests session with the authorization header.
SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "accept": "application/json"
})

# --------------------------------------------------------------------------
# ## Core Helper Function
# --------------------------------------------------------------------------

def _make_request(
    method: str, 
    endpoint: str, 
    response_model: Type[BaseModel],
    payload_json: Optional[str] = None # Expect a JSON string for POST
) -> Optional[List[BaseModel]]:
    """A helper function to make requests, validate, and parse the response."""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method.upper() == 'GET':
            response = SESSION.get(url)
        else: # POST
            response = SESSION.post(url, data=payload_json)

        response.raise_for_status()
        response_json = response.json()

        if response_json.get("output", {}).get("type") == "success":
            data = response_json.get("output", {}).get("data")
            # Use Pydantic's TypeAdapter to validate the list of objects
            adapter = TypeAdapter(List[response_model])
            return adapter.validate_python(data)
        else:
            error_message = response_json.get("output", {}).get("message", "Unknown API error")
            print(f"API returned an error: {error_message}")
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {http_err.response.text}")
    except ValidationError as e:
        print(f"Pydantic validation error: Failed to parse response from {endpoint}.\n{e}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
    
    return None

# --------------------------------------------------------------------------
# ## API Functions
# --------------------------------------------------------------------------

def get_commodities() -> Optional[List[Commodity]]:
    """Fetches and validates the list of all commodities."""
    return _make_request("GET", "/agmarknet/commodities", response_model=Commodity)

def get_geographies() -> Optional[List[Geography]]:
    """Fetches and validates the list of all states and districts."""
    return _make_request("GET", "/agmarknet/geographies", response_model=Geography)

def get_states() -> Optional[List[State]]:
    """Extracts a unique list of all available states."""
    geographies = get_geographies()
    if not geographies:
        return None
    
    states_set = {(geo.census_state_id, geo.census_state_name) for geo in geographies}
    
    return sorted(
        [State(state_id=sid, state_name=sname) for sid, sname in states_set],
        key=lambda x: x.state_id
    )

def get_districts(state_id: int) -> Optional[List[District]]:
    """Extracts the list of districts for a given state ID."""
    geographies = get_geographies()
    if not geographies:
        return None
        
    districts = []
    for geo in geographies:
        if geo.census_state_id == state_id:
            districts.append(District(
                district_id=geo.census_district_id,
                district_name=geo.census_district_name
            ))
            
    if not districts:
        print(f"State with ID {state_id} not found or has no districts.")
        return None
        
    unique_districts = {(d.district_id, d.district_name) for d in districts}
    return sorted(
        [District(district_id=did, district_name=dname) for did, dname in unique_districts],
        key=lambda x: x.district_id
    )

def get_prices(payload: PricesRequest) -> Optional[List[PriceData]]:
    """Fetches and validates price data."""
    # Use Pydantic's model_dump_json to correctly serialize dates
    payload_json_str = payload.model_dump_json(exclude_unset=True)
    return _make_request("POST", "/agmarknet/prices", payload_json=payload_json_str, response_model=PriceData)

def get_quantities(payload: QuantitiesRequest) -> Optional[List[QuantityData]]:
    """Fetches and validates arrival quantity data."""
    payload_json_str = payload.model_dump_json(exclude_unset=True)
    return _make_request("POST", "/agmarknet/quantities", payload_json=payload_json_str, response_model=QuantityData)

# --------------------------------------------------------------------------
# ## Example Usage
# --------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"DEBUG: Using API Key starting with '{API_KEY[:5]}...'\n")

    # 1. Get all commodities and find "Wheat"
    print("--- Fetching Commodities ---")
    commodities = get_commodities()
    print(commodities)
    if commodities:
        print(f"Successfully fetched and validated {len(commodities)} commodities.")
        wheat_commodity = next((c for c in commodities if c.commodity_name == 'Wheat'), None)
        if wheat_commodity:
            print(f"Found 'Wheat' with ID: {wheat_commodity.commodity_id}\n")
        else:
            print("Could not find 'Wheat'. Using fallback.\n")
            wheat_commodity = Commodity(commodity_id=1, commodity_name='Wheat') # Fallback
    
    # 2. Get all states and find "Uttar Pradesh"
    print("--- Fetching States ---")
    states = get_states()
    print(states)
    if states:
        print(f"Successfully fetched and validated {len(states)} states.")
        up_state = next((s for s in states if s.state_name == 'Uttar Pradesh'), None)
        if up_state:
            print(f"Found 'Uttar Pradesh' with ID: {up_state.state_id}\n")
        else:
            print("Could not find 'Uttar Pradesh'. Using fallback.\n")
            up_state = State(state_id=9, state_name='Uttar Pradesh') # Fallback
    
    # 3. Get districts for "Uttar Pradesh" and find "Agra"
    print(f"--- Fetching Districts for Uttar Pradesh (ID: {up_state.state_id}) ---")
    districts = get_districts(state_id=up_state.state_id)
    print(districts)
    if districts:
        print(f"Successfully fetched and validated {len(districts)} districts.")
        agra_district = next((d for d in districts if d.district_name == 'Agra'), None)
        if agra_district:
            print(f"Found 'Agra' with ID: {agra_district.district_id}\n")
        else:
            print("Could not find 'Agra'. Using fallback.\n")
            agra_district = District(district_id=104, district_name='Agra') # Fallback

    # 4. Get prices for Wheat in Agra, UP for a specific date range
    print(f"--- Fetching Prices for Wheat in Agra, UP ---")
    price_request_payload = PricesRequest(
        commodity_id=wheat_commodity.commodity_id,
        state_id=up_state.state_id,
        district_id=[agra_district.district_id], # Must be a list
        from_date=date(2025, 4, 1),
        to_date=date(2025, 4, 7)
    )
    price_data = get_prices(price_request_payload)
    if price_data:
        print(f"Successfully fetched and validated {len(price_data)} price records.")
        # Access data using Pydantic model attributes
        print(f"Sample price data for {price_data[0].date}: Modal Price = {price_data[0].modal_price}")
        print(price_data)
    else:
        print("Could not fetch price data.")
    print("-" * 40 + "\n")

    # 5. Get quantities for the same filters
    print(f"--- Fetching Quantities for Wheat in Agra, UP ---")
    quantity_request_payload = QuantitiesRequest(
        commodity_id=wheat_commodity.commodity_id,
        state_id=up_state.state_id,
        district_id=[agra_district.district_id], # Must be a list
        from_date=date(2025, 4, 1),
        to_date=date(2025, 4, 7)
    )
    quantity_data = get_quantities(quantity_request_payload)
    if quantity_data:
        print(f"Successfully fetched and validated {len(quantity_data)} quantity records.")
        print(f"Sample quantity data for {quantity_data[0].date}: Quantity = {quantity_data[0].quantity}")
        print(quantity_data)
    else:
        print("Could not fetch quantity data.")