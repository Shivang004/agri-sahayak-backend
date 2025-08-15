# graph.py
import json
from typing import TypedDict, Annotated, Dict, Optional, List
from langgraph.graph import StateGraph, END
from datetime import date, timedelta
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from thefuzz import process
# Import agents and helper functions
from agents import (
    sarpanch_agent, planner_agent, general_agent, response_agent,
    agent_vayu, agent_bhumi, agent_dr_fasal, agent_bazar, agent_yojana
)
from agno.media import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tools.market_tools import get_prices, get_quantities
from tools.weather_tools import get_comprehensive_agri_report
from models import (    PricesRequest, QuantitiesRequest,
)

def get_price_plot_base64(
    commodity_id: int,
    state_id: int,
    district_id: List[int],
    to_date: date,
    days: int = 60,
) -> Optional[str]:
    """
    Fetches 60-day price data, generates a plot of the daily average
    modal price, saves it as a PNG, and returns it as Base64.
    """
    from_date = to_date - timedelta(days=days - 1)
    price_request = PricesRequest(
        commodity_id=commodity_id,
        state_id=state_id,
        district_id=district_id,
        from_date=from_date,
        to_date=to_date,
    )
    price_data = get_prices(price_request)

    if not price_data:
        print("No price data found for the given filters.")
        return None

    df = pd.DataFrame([p.model_dump() for p in price_data])
    df["date"] = pd.to_datetime(df["date"])

    daily_prices = df.groupby("date").agg(
        modal_price=('modal_price', 'mean')
    ).reset_index()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(
        data=daily_prices, x="date", y="modal_price", ax=ax,
        label="Average Modal Price", color="#0077b6", linewidth=2.5, marker='o'
    )

    # Add latest price label
    latest_price = daily_prices['modal_price'].iloc[-1]
    ax.text(0.02, 0.85, f'Latest Price: ₹{latest_price:.2f}', 
            transform=ax.transAxes, fontsize=14, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top')

    ax.set_title(f"Daily Modal Price Trend ({days} Days)", fontsize=16, weight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (INR)", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # --- SAVE AND CONVERT PLOT ---
    # Save the figure to a file
    # fig.savefig("daily_modal_price_trend.png", dpi=90, bbox_inches='tight')

    # Convert plot image to a Base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    print("✅ Price plot saved to daily_modal_price_trend.png")
    return img_base64


def get_quantity_plot_base64(
    commodity_id: int,
    state_id: int,
    district_id: List[int],
    to_date: date,
    days: int = 60,
) -> Optional[str]:
    """
    Fetches 60-day arrival quantity data, generates a line plot,
    saves it as a PNG, and returns it as Base64.
    """
    from_date = to_date - timedelta(days=days - 1)
    quantity_request = QuantitiesRequest(
        commodity_id=commodity_id,
        state_id=state_id,
        district_id=district_id,
        from_date=from_date,
        to_date=to_date,
    )
    quantity_data = get_quantities(quantity_request)

    if not quantity_data:
        print("No quantity data found for the given filters.")
        return None

    df = pd.DataFrame([q.model_dump() for q in quantity_data])
    df["date"] = pd.to_datetime(df["date"])

    daily_quantities = df.groupby("date").agg(
        quantity=('quantity', 'sum')
    ).reset_index()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(
        data=daily_quantities, x="date", y="quantity", ax=ax,
        label="Total Daily Quantity", color="#2a9d8f", linewidth=2.5, marker='o'
    )

    # Add latest quantity label
    latest_quantity = daily_quantities['quantity'].iloc[-1]
    ax.text(0.02, 0.85, f'Latest Quantity: {latest_quantity:.1f} Tonnes', 
            transform=ax.transAxes, fontsize=14, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top')

    ax.set_title(f"Daily Arrival Quantity ({days} Days)", fontsize=16, weight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Total Quantity (Tonnes)", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # --- SAVE AND CONVERT PLOT ---
    # Save the figure to a file
    # fig.savefig("daily_quantity_trend.png", dpi=90, bbox_inches='tight')

    # Convert plot image to a Base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    print("✅ Quantity plot saved to daily_quantity_trend.png")
    return img_base64


COMMODITY_DATA = []

def load_commodity_data(file_path: str = 'commodity.json'):
    """Loads commodity data from a JSON file."""
    global COMMODITY_DATA
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            COMMODITY_DATA = json.load(f)
        print(f"Successfully loaded {len(COMMODITY_DATA)} commodities.")
    except Exception as e:
        print(f"Error loading commodity data: {e}")
        COMMODITY_DATA = []

def find_commodity_id(crop_name: str) -> Optional[int]:
    """Uses fuzzy matching to find the best commodity_id for a crop name."""
    if not COMMODITY_DATA or not crop_name:
        return None
    commodity_names = {item["commodity_name"]: item["commodity_id"] for item in COMMODITY_DATA}
    best_match = process.extractOne(crop_name.lower(), commodity_names.keys())
    if best_match and best_match[1] > 80:
        print(f"Fuzzy Match: '{best_match[0]}' for '{crop_name}' with score {best_match[1]}")
        return commodity_names[best_match[0]]
    print(f"No confident commodity match found for '{crop_name}'")
    return None

# Load data on script initialization
load_commodity_data()
# --- State Definition for the Graph ---
class AgentState(TypedDict):
    original_query: str
    target_language: str
    user_context: dict # For lat, lon, state_id, district_id
    image_url: Optional[str] #base64 string
    query_en: str
    mode: str
    planner_output: dict
    agent_outputs: dict
    final_response: str

# --- Node Definitions ---

def run_sarpanch(state: AgentState):
    """Runs the Sarpanch agent to process the initial query."""
    print("---NODE: SARPANCH---")
    print(f"Input query: {state['original_query']}")
    print(f"Target language: {state['target_language']}")
    
    response = sarpanch_agent.run(state['original_query'])
    print(f"Sarpanch agent response.content: {response.content}")
    
    # Clean the response content by removing markdown code blocks
    cleaned_content = response.content.strip()
    if cleaned_content.startswith('```json'):
        cleaned_content = cleaned_content[7:]  # Remove ```json
    if cleaned_content.startswith('```'):
        cleaned_content = cleaned_content[3:]  # Remove ```
    if cleaned_content.endswith('```'):
        cleaned_content = cleaned_content[:-3]  # Remove ```
    
    cleaned_content = cleaned_content.strip()
    print(f"Cleaned content: {cleaned_content}")
    
    try:
        result = json.loads(cleaned_content)
        state['mode'] = result['mode']
        state['query_en'] = result['query_en']
        print(f"Sarpanch Output: Mode='{state['mode']}', Query='{state['query_en']}'")
        print(f"Parsed result: {result}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing Sarpanch output: {e}. Defaulting to multi-mode.")
        raise e
    return state

def route_by_mode(state: AgentState):
    """Decides the next step based on the mode set by Sarpanch."""
    print(f"---ROUTER: Mode is '{state['mode']}'---")
    print(f"Current state keys: {list(state.keys())}")
    print(f"Mode value: {state['mode']}")
    print(f"Query_en value: {state['query_en']}")
    
    if state['mode'] == 'single':
        print("Routing to: general_agent")
        return 'general_agent'
    else:
        print("Routing to: planner")
        return 'planner'

def run_planner(state: AgentState):
    """Runs the Planner agent to create a multi-agent execution plan."""
    print("---NODE: PLANNER---")
    print(f"Input query_en: {state['query_en']}")
    
    response = planner_agent.run(state['query_en'])
    print(f"Planner agent response.content: {response.content}")
    
    # Clean the response content by removing markdown code blocks
    cleaned_content = response.content.strip()
    if cleaned_content.startswith('```json'):
        cleaned_content = cleaned_content[7:]  # Remove ```json
    if cleaned_content.startswith('```'):
        cleaned_content = cleaned_content[3:]  # Remove ```
    if cleaned_content.endswith('```'):
        cleaned_content = cleaned_content[:-3]  # Remove ```
    
    cleaned_content = cleaned_content.strip()
    print(f"Cleaned content: {cleaned_content}")
    
    try:
        result = json.loads(cleaned_content)
        state['planner_output'] = result
        print(f"Planner Output: {result}")
        print(f"Planner agents: {result.get('agents', {})}")
        print(f"Planner tasks: {result.get('tasks', {})}")
        print(f"Planner metadata: {result.get('metadata', {})}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing Planner output: {e}. Cannot proceed with multi-agent plan.")
        # Handle error case, maybe by setting a flag or a default empty plan
        # state['planner_output'] = {"agents": {}, "tasks": {}, "metadata": {}}
        raise e
    return state

def run_general_agent(state: AgentState):
    """Runs the General Agriculture agent for simple queries."""
    print("---NODE: GENERAL AGENT---")
    print(f"Input query_en: {state['query_en']}")
    
    response = general_agent.run(state['query_en'])
    print(f"General agent response.content: {response.content}")
    
    state['agent_outputs'] = {"general_knowledge": response.content}
    print(f"Agent outputs set: {list(state['agent_outputs'].keys())}")
    return state

def run_domain_agents(state: AgentState):
    """Parameter Resolver and parallel execution of domain agents."""
    print("---NODE: DOMAIN AGENTS---")
    plan = state['planner_output']
    user_ctx = state['user_context']
    outputs = {}
    
    print(f"Planner output: {plan}")
    print(f"User context: {user_ctx}")
    print(f"Image URL present: {state['image_url'] is not None}")

    # This part can be parallelized for production using ThreadPoolExecutor
    for agent_name, is_active in plan.get('agents', {}).items():
        print(f"Checking agent: {agent_name}, is_active: {is_active}")
        if is_active:
            task = plan.get('tasks', {}).get(agent_name)
            if not task:
                print(f"No task found for agent {agent_name}, skipping")
                continue

            print(f"  > Running agent: {agent_name.upper()} for task: '{task}'")
            
            # --- Parameter Resolution ---
            if agent_name == 'vayu':
                print(f"Running Vayu agent with weather data...")
                weather_report = get_comprehensive_agri_report(user_ctx['lat'], user_ctx['lon'])
                print(f"Weather report obtained: {len(weather_report)} characters")
                full_task = str(task) + f"Below is the weather report for your location: \n {weather_report}"
                response = agent_vayu.run(full_task)
                print(f"Vayu agent response.content: {response.content}")
                outputs['vayu'] = response.content
            
            elif agent_name == 'bazar':
                print(f"Running Bazar agent...")
                crop_names = plan.get('metadata', {}).get('crop_names', [])
                print(f"Crop names from metadata: {crop_names}")
                if crop_names:
                    crop_name = crop_names[0] # Assuming one crop for simplicity
                    print(f"Processing crop: {crop_name}")
                    commodity_id = find_commodity_id(crop_name)
                    print(f"Found commodity_id: {commodity_id}")
                    if commodity_id:
                        today = date.today()
                        print(f"Fetching price data for commodity_id: {commodity_id}")
                        price_data_img = get_price_plot_base64(
                            commodity_id=commodity_id,
                            state_id=user_ctx['state_id'],
                            district_id=user_ctx['district_id'],
                            to_date=today,
                            days=60)
                        print(f"Price data image generated: {price_data_img is not None}")
                        
                        print(f"Fetching quantity data for commodity_id: {commodity_id}")
                        quantity_data_img = get_quantity_plot_base64(
                            commodity_id=commodity_id,
                            state_id=user_ctx['state_id'],
                            district_id=user_ctx['district_id'],
                            to_date=today,
                            days=60)
                        print(f"Quantity data image generated: {quantity_data_img is not None}")
                        
                        # Decode Base64 strings back to bytes for the Image objects
                        price_bytes = base64.b64decode(price_data_img) if price_data_img else None
                        quantity_bytes = base64.b64decode(quantity_data_img) if quantity_data_img else None
                        
                        images = []
                        if price_bytes:
                            images.append(Image(content=price_bytes))
                        if quantity_bytes:
                            images.append(Image(content=quantity_bytes))
                        
                        response = agent_bazar.run(task, images=images)
                        print(f"Bazar agent response.content: {response.content}")
                        outputs['bazar'] = response.content
                    else:
                        error_msg = f"Could not find a market ID for the crop '{crop_name}'."
                        print(error_msg)
                        outputs['bazar'] = error_msg
                else:
                    print("No crop names found in metadata")
            
            elif agent_name == 'dr_fasal':
                print(f"Running Dr. Fasal agent...")
                print(f"Image URL present: {state['image_url'] is not None}")
                # Dr. Fasal can directly use the image if provided in the run call
                images = None
                if state['image_url']:
                    try:
                        # Decode Base64 string back to bytes for the Image object
                        image_bytes = base64.b64decode(state['image_url'])
                        print(f"Decoded image bytes length: {len(image_bytes)}")
                        images = [Image(content=image_bytes)]
                    except Exception as e:
                        print(f"Error decoding image: {e}")
                        images = None
                
                response = agent_dr_fasal.run(task, images=images)
                print(f"Dr. Fasal agent response.content: {response.content}")
                outputs['dr_fasal'] = response.content

            elif agent_name == 'bhumi':
                print(f"Running Bhumi agent...")
                response = agent_bhumi.run(task)
                print(f"Bhumi agent response.content: {response.content}")
                outputs['bhumi'] = response.content

            elif agent_name == 'yojana':
                print(f"Running Yojana agent...")
                response = agent_yojana.run(task)
                print(f"Yojana agent response.content: {response.content}")
                outputs['yojana'] = response.content
    
    print(f"All domain agents completed. Outputs: {list(outputs.keys())}")
    state['agent_outputs'] = outputs
    return state

def run_response_agent(state: AgentState):
    """Aggregates results and generates the final user-facing response."""
    print("---NODE: RESPONSE AGENT---")
    print(f"Target language: {state['target_language']}")
    print(f"Query_en: {state['query_en']}")
    print(f"Agent outputs keys: {list(state['agent_outputs'].keys())}")
    
    context_str = f"Target Language: {state['target_language']}\n\n"
    context_str += f"Original Question (English): {state['query_en']}\n\n"
    context_str += "--- Expert Reports ---\n"
    
    for agent, output in state['agent_outputs'].items():
        print(f"Processing agent: {agent}")
        print(f"Agent output length: {len(str(output))} characters")
        context_str += f"## {agent.replace('_', ' ').title()} Report:\n{output}\n\n"

    print(f"Context string length: {len(context_str)} characters")
    print(f"Context string preview: {context_str[:500]}...")

    response = response_agent.run(context_str)
    print(f"Response agent response.content: {response.content}")
    
    state['final_response'] = response.content
    print(f"Final Response Generated for language '{state['target_language']}'")
    print(f"Final response length: {len(response.content)} characters")
    return state


# --- Build the Graph ---
def build_graph():
    print("=== BUILDING GRAPH ===")
    workflow = StateGraph(AgentState)
    print("StateGraph created")

    # Add nodes
    print("Adding nodes...")
    workflow.add_node("sarpanch", run_sarpanch)
    print("Added sarpanch node")
    workflow.add_node("planner", run_planner)
    print("Added planner node")
    workflow.add_node("general_agent", run_general_agent)
    print("Added general_agent node")
    workflow.add_node("domain_agents", run_domain_agents)
    print("Added domain_agents node")
    workflow.add_node("response_agent", run_response_agent)
    print("Added response_agent node")

    # Set entry point
    print("Setting entry point...")
    workflow.set_entry_point("sarpanch")
    print("Entry point set to sarpanch")

    # Add edges
    print("Adding edges...")
    workflow.add_conditional_edges(
        "sarpanch",
        route_by_mode,
        {"general_agent": "general_agent", "planner": "planner"}
    )
    print("Added conditional edges from sarpanch")
    workflow.add_edge("planner", "domain_agents")
    print("Added edge: planner -> domain_agents")
    workflow.add_edge("general_agent", "response_agent")
    print("Added edge: general_agent -> response_agent")
    workflow.add_edge("domain_agents", "response_agent")
    print("Added edge: domain_agents -> response_agent")
    workflow.add_edge("response_agent", END)
    print("Added edge: response_agent -> END")
    
    # Compile the graph
    print("Compiling graph...")
    app = workflow.compile()
    print("Graph compiled successfully")
    
    # Optional: Visualize the graph
    # try:
    #     print("Generating graph visualization...")
    #     img = app.get_graph().draw_mermaid_png()
    #     with open("graph.png", "wb") as f:
    #         f.write(img)
    #     print("Graph visualization saved to graph.png")
    # except Exception as e:
    #     print(f"Could not draw graph: {e}")

    print("=== GRAPH BUILDING COMPLETED ===")
    return app

def main():
    """
    Main function to run a test case through the agent graph.
    This test simulates a user query involving a crop disease image,
    market prices for that crop, and related weather advice.
    """
    print("=== STARTING MAIN FUNCTION ===")
    
    # 1. Build the graph
    print("Building graph...")
    app = build_graph()
    print("Graph built successfully")

    # 2. Define the initial state for the test case
    #    - query: A complex query in a non-English language.
    #    - user_context: Location data required by weather and market agents.
    #    - image_path: Path to a local image of a diseased plant.
    #    - target_language: The language for the final response.

    query = "Genhu ka अगले कुछ हफ्तों में बाजार का रुझान क्या है और मौसम कैसा रहेगा?"
    # query = "मेरी Genhu की फसल में यह बीमारी है, मुझे क्या करना चाहिए? अगले कुछ हफ्तों में बाजार का रुझान क्या है और मौसम कैसा रहेगा?"
    # query  = "मेरी Genhu की फसल में यह बीमारी है, Pata Lagayein Or Sujhav de मुझे क्या करना चाहिए? Konse Keet Nashak Lgana chahiye? "
    user_context = {
        "lat": 26.4499,
        "lon": 80.3319,
        "state_id": 8,      # Rajasthan
        "district_id": [104] # Alwar
    }
    image_path = "disease.png" # IMPORTANT: Create a dummy file named 'disease.png' or use a real image path
    target_language = "hindi"

    print(f"Query: {query}")
    print(f"User context: {user_context}")
    print(f"Target language: {target_language}")

    # 3. Encode the image to a Base64 string
    image_base64 = None
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        print(f"Successfully encoded image: {image_path}")
        print(f"Image base64 length: {len(image_base64)} characters")
    except FileNotFoundError:
        print(f"Warning: Image file not found at '{image_path}'. Dr. Fasal agent will not receive an image.")
    except Exception as e:
        print(f"An error occurred while encoding the image: {e}")


    # 4. Construct the initial state dictionary
    initial_state = {
        "original_query": query,
        "target_language": target_language,
        "user_context": user_context,
        "image_url": image_base64, # Pass the base64 string
        "agent_outputs": {},
    }
    
    print(f"Initial state keys: {list(initial_state.keys())}")
    print(f"Initial state prepared successfully")

    # 5. Run the graph with the initial state
    print("\n--- EXECUTING AGENT GRAPH ---")
    final_state = app.invoke(initial_state)
    print("Graph execution completed")

    # 6. Print the final response
    print("\n--- FINAL RESPONSE ---")
    final_response = final_state.get("final_response", "No final response generated.")
    print(final_response)
    print("----------------------\n")
    
    # 7. Print final state summary
    print("=== FINAL STATE SUMMARY ===")
    print(f"Final state keys: {list(final_state.keys())}")
    for key, value in final_state.items():
        if isinstance(value, str):
            print(f"{key}: {len(value)} characters")
        elif isinstance(value, dict):
            print(f"{key}: {list(value.keys())}")
        else:
            print(f"{key}: {type(value)}")
    print("=== END MAIN FUNCTION ===")


if __name__ == "__main__":
    # To run the test, execute this file from your terminal:
    # python team.py
    main()
