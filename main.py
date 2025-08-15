from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uvicorn
from dotenv import load_dotenv

# Import our modules
from models import QueryRequest, QueryResponse, ErrorResponse
from team import build_graph
from routers import data_endpoints

load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Agri-Sahayak Backend",
    description="Multi-agent agricultural assistance system powered by Agno framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include data endpoints router
app.include_router(data_endpoints.router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Welcome to Agri-Sahayak Backend",
        "version": "1.0.0",
        "description": "Multi-agent agricultural assistance system",
        "endpoints": {
            "conversational_ai": "/api/query",
            "market_data": "/api/data/prices",
            "weather_data": "/api/data/weather",
            "commodities": "/api/data/commodities",
            "documentation": "/docs"
        }
    }


@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    query: str = Form(...),
    language: str = Form(default="en"),
    latitude: float = Form(...),
    longitude: float = Form(...),
    state_id: int = Form(...),
    district_id: str = Form(...),  # Will be parsed as JSON string
    image: UploadFile = File(default=None)
):
    """
    Process a farmer's query using the multi-agent team
    
    This endpoint receives transcribed text from the frontend and routes it
    through the appropriate specialist agents to provide comprehensive answers.
    """
    try:
        # Validate input
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Parse district_id from JSON string
        import json
        try:
            district_ids = json.loads(district_id)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid district_id format")
        
        app = build_graph()  # Build the agent team graph
        
        user_context = {
            "lat": latitude,
            "lon": longitude,
            "state_id": state_id,
            "district_id": district_ids,
        }
        
        # Handle image upload
        image_data = None
        if image:
            image_data = await image.read()
            # Convert to base64 for the team.py processing
            import base64
            image_data = base64.b64encode(image_data).decode('utf-8')
            print(f"Image base64 length: {len(image_data)} characters")
        target_language = language

        print(f"Query: {query}")
        print(f"User context: {user_context}")
        print(f"Target language: {target_language}")
        print(f"Image present: {image_data is not None}")

        # Build the agent team graph       
        initial_state = {
            "original_query": query.strip(),
            "target_language": target_language,
            "user_context": user_context,
            "image_url": image_data,  # Pass the base64 string
            "agent_outputs": {},
        }
    
        print(f"Initial state keys: {list(initial_state.keys())}")
        print(f"Initial state prepared successfully")

        # 5. Run the graph with the initial state
        print("\n--- EXECUTING AGENT GRAPH ---")
        final_state = app.invoke(initial_state)
        print("Graph execution completed")

        # 6. Extract response data from final state
        print("\n--- FINAL RESPONSE ---")
        final_response = final_state.get("final_response", "No final response generated.")
        agent_outputs = final_state.get("agent_outputs", {})
        mode = final_state.get("mode", "unknown")
        query_en = final_state.get("query_en", query)
        agents_used = list(agent_outputs.keys()) if agent_outputs else []
        
        print(f"Final response: {final_response}")
        print(f"Agents used: {agents_used}")
        print("----------------------\n")
        
        return QueryResponse(
            response=final_response,
            agents_used=agents_used,
            agent_outputs=agent_outputs,
            mode=mode,
            query_en=query_en
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": "The requested endpoint does not exist"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    print(f"Starting Agri-Sahayak Backend on {host}:{port}")
    print(f"Debug mode: {debug}")
    print(f"Documentation available at: http://{host}:{port}/docs")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
