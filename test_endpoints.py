import requests
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, datetime, timedelta
import seaborn as sns
from typing import List, Dict, Any, Optional
import numpy as np

# Configuration
BASE_URL = "http://localhost:8000"  # Adjust if your server runs on a different port
API_BASE = f"{BASE_URL}/api/data"

# Set up matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EndpointTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/data"
        
    def test_health_check(self) -> bool:
        """Test the health check endpoint"""
        try:
            response = requests.get(f"{self.api_base}/health")
            if response.status_code == 200:
                print("✅ Health check passed")
                print(f"Response: {response.json()}")
                return True
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_commodities(self) -> Optional[List[Dict]]:
        """Test the commodities endpoint"""
        try:
            response = requests.get(f"{self.api_base}/commodities")
            if response.status_code == 200:
                data = response.json()
                commodities = data.get("commodities", [])
                print(f"✅ Commodities endpoint passed - Found {len(commodities)} commodities")
                print(f"Sample commodities: {commodities[:3]}")
                return commodities
            else:
                print(f"❌ Commodities endpoint failed with status {response.status_code}")
                print(f"Error: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Commodities endpoint error: {e}")
            return None
    
    def test_states(self) -> Optional[List[Dict]]:
        """Test the states endpoint"""
        try:
            response = requests.get(f"{self.api_base}/states")
            if response.status_code == 200:
                data = response.json()
                states = data.get("states", [])
                print(f"✅ States endpoint passed - Found {len(states)} states")
                print(f"Sample states: {states[:3]}")
                return states
            else:
                print(f"❌ States endpoint failed with status {response.status_code}")
                print(f"Error: {response.text}")
                return None
        except Exception as e:
            print(f"❌ States endpoint error: {e}")
            return None
    
    def test_districts(self, state_id: int) -> Optional[List[Dict]]:
        """Test the districts endpoint for a specific state"""
        try:
            response = requests.get(f"{self.api_base}/districts/{state_id}")
            if response.status_code == 200:
                data = response.json()
                districts = data.get("districts", [])
                print(f"✅ Districts endpoint passed for state {state_id} - Found {len(districts)} districts")
                print(f"Sample districts: {districts[:3]}")
                return districts
            else:
                print(f"❌ Districts endpoint failed with status {response.status_code}")
                print(f"Error: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Districts endpoint error: {e}")
            return None
    
    def test_prices(self, commodity_id: int, state_id: int, district_id: int, 
                   from_date: str, to_date: str) -> Optional[List[Dict]]:
        """Test the prices endpoint"""
        try:
            payload = {
                "commodity_id": commodity_id,
                "from_date": from_date,
                "to_date": to_date,
                "state_id": state_id,
                "district_id": [district_id]
            }
            
            response = requests.post(f"{self.api_base}/prices", json=payload)
            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices", [])
                print(f"✅ Prices endpoint passed - Found {len(prices)} price records")
                if prices:
                    print(f"Sample price data: {prices[0]}")
                return prices
            else:
                print(f"❌ Prices endpoint failed with status {response.status_code}")
                print(f"Error: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Prices endpoint error: {e}")
            return None
    
    def test_quantities(self, commodity_id: int, state_id: int, district_id: int,
                       from_date: str, to_date: str) -> Optional[List[Dict]]:
        """Test the quantities endpoint"""
        try:
            payload = {
                "commodity_id": commodity_id,
                "from_date": from_date,
                "to_date": to_date,
                "state_id": state_id,
                "district_id": [district_id]
            }
            
            response = requests.post(f"{self.api_base}/quantities", json=payload)
            if response.status_code == 200:
                data = response.json()
                quantities = data.get("quantities", [])
                print(f"✅ Quantities endpoint passed - Found {len(quantities)} quantity records")
                if quantities:
                    print(f"Sample quantity data: {quantities[0]}")
                return quantities
            else:
                print(f"❌ Quantities endpoint failed with status {response.status_code}")
                print(f"Error: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Quantities endpoint error: {e}")
            return None
    
    def run_all_tests(self):
        """Run all endpoint tests"""
        print("🚀 Starting comprehensive endpoint tests...\n")
        
        # Test health check
        self.test_health_check()
        print()
        
        # Test commodities
        commodities = self.test_commodities()
        print()
        
        # Test states
        states = self.test_states()
        print()
        
        # Test districts for a sample state (assuming state_id 9 is Uttar Pradesh)
        if states:
            sample_state = {"state_id": 8}
            districts = self.test_districts(sample_state["state_id"])
            print()
            
            # Test prices and quantities with sample data
            if commodities and districts:
                sample_commodity = {"commodity_id": 1}
                sample_district = {"district_id": 104}
                
                # Use recent dates for testing
                from_date = (date.today() - timedelta(days=60)).isoformat()
                to_date = date.today().isoformat()
                
                prices = self.test_prices(
                    sample_commodity["commodity_id"],
                    sample_state["state_id"],
                    sample_district["district_id"],
                    from_date,
                    to_date
                )
                print()
                
                quantities = self.test_quantities(
                    sample_commodity["commodity_id"],
                    sample_state["state_id"],
                    sample_district["district_id"],
                    from_date,
                    to_date
                )
                print()
                
                # Return data for visualization
                return {
                    "commodities": commodities,
                    "states": states,
                    "districts": districts,
                    "prices": prices,
                    "quantities": quantities
                }
        
        return None


class DataVisualizer:
    def __init__(self):
        self.colors = {
            'min_price': '#FF6B6B',
            'max_price': '#4ECDC4',
            'modal_price': '#45B7D1',
            'quantity': '#96CEB4'
        }
    
    def plot_prices(self, price_data: List[Dict], commodity_name: str = "Commodity", 
                   location: str = "Location", save_path: str = "price_analysis.png"):
        """Plot min, max, and modal prices in one line graph"""
        if not price_data:
            print("❌ No price data available for plotting")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(price_data)
        
        # Convert date strings to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        
        # --- START: NEW AGGREGATION CODE ---
        # Group by date and aggregate the price data
        # Get the lowest min_price, highest max_price, and average modal_price for each day
        daily_prices = df.groupby('date').agg(
            min_price=('min_price', 'min'),
            max_price=('max_price', 'max'),
            modal_price=('modal_price', 'mean')
        ).reset_index()
        # --- END: NEW AGGREGATION CODE ---
        
        # Sort by date
        daily_prices = daily_prices.sort_values('date')
        
        # Create single line plot
        plt.figure(figsize=(12, 6))
        
        # Plot the aggregated data
        plt.plot(daily_prices['date'], daily_prices['min_price'], label='Min Price', color=self.colors['min_price'], linewidth=2, marker='o')
        plt.plot(daily_prices['date'], daily_prices['max_price'], label='Max Price', color=self.colors['max_price'], linewidth=2, marker='s')
        plt.plot(daily_prices['date'], daily_prices['modal_price'], label='Modal Price', color=self.colors['modal_price'], linewidth=2, marker='^')
        
        plt.title(f'Price Analysis for {commodity_name} in {location}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (₹/Quintal)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Price analysis saved to {save_path}")
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 Price Summary Statistics for {commodity_name} in {location}:")
        print(f"Min Price Range: ₹{df['min_price'].min():.2f} - ₹{df['min_price'].max():.2f}")
        print(f"Max Price Range: ₹{df['max_price'].min():.2f} - ₹{df['max_price'].max():.2f}")
        print(f"Modal Price Range: ₹{df['modal_price'].min():.2f} - ₹{df['modal_price'].max():.2f}")
        print(f"Average Price Spread: ₹{((df['max_price'] - df['min_price']).mean()):.2f}")
    
    def plot_quantities(self, quantity_data: List[Dict], commodity_name: str = "Commodity", 
                       location: str = "Location", save_path: str = "quantity_analysis.png"):
        """Plot quantity data as a single line plot"""
        if not quantity_data:
            print("❌ No quantity data available for plotting")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(quantity_data)
        
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'])

        # --- START: NEW AGGREGATION CODE ---
        # Group by date and sum the quantities to get the total daily arrival
        daily_quantities = df.groupby('date').agg(
            quantity=('quantity', 'sum')
        ).reset_index()
        # --- END: NEW AGGREGATION CODE ---
        
        # Sort by date
        daily_quantities = daily_quantities.sort_values('date')
        
        # Create single line plot
        plt.figure(figsize=(12, 6))
        
        # Plot the aggregated quantity trends over time
        plt.plot(daily_quantities['date'], daily_quantities['quantity'], label='Arrival Quantity', 
                color=self.colors['quantity'], linewidth=3, marker='o')
        
        plt.title(f'Quantity Analysis for {commodity_name} in {location}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Quantity (Tonnes)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Quantity analysis saved to {save_path}")
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 Quantity Summary Statistics for {commodity_name} in {location}:")
        print(f"Total Quantity: {df['quantity'].sum():.2f} tonnes")
        print(f"Average Daily Quantity: {df['quantity'].mean():.2f} tonnes")
        print(f"Quantity Range: {df['quantity'].min():.2f} - {df['quantity'].max():.2f} tonnes")
        print(f"Standard Deviation: {df['quantity'].std():.2f} tonnes")
    
    def plot_combined_analysis(self, price_data: List[Dict], quantity_data: List[Dict], 
                              commodity_name: str = "Commodity", location: str = "Location",
                              save_path: str = "combined_analysis.png"):
        """Plot combined price and quantity analysis as 2 line plots"""
        if not price_data or not quantity_data:
            print("❌ Both price and quantity data required for combined analysis")
            return
        
        # Convert to DataFrames
        price_df = pd.DataFrame(price_data)
        quantity_df = pd.DataFrame(quantity_data)
        
        # Convert date strings to datetime
        price_df['date'] = pd.to_datetime(price_df['date'])
        quantity_df['date'] = pd.to_datetime(quantity_df['date'])
        
        # Merge data on date
        combined_df = pd.merge(price_df, quantity_df, on='date', how='inner', suffixes=('_price', '_quantity'))
        
        if combined_df.empty:
            print("❌ No overlapping dates found for combined analysis")
            return
        
        # Create 2 line plots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Price trends
        ax1.plot(combined_df['date'], combined_df['min_price'], label='Min Price', 
                color=self.colors['min_price'], linewidth=2, marker='o')
        ax1.plot(combined_df['date'], combined_df['max_price'], label='Max Price', 
                color=self.colors['max_price'], linewidth=2, marker='s')
        ax1.plot(combined_df['date'], combined_df['modal_price'], label='Modal Price', 
                color=self.colors['modal_price'], linewidth=2, marker='^')
        
        ax1.set_title('Price Trends', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (₹/Quintal)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Quantity trends
        ax2.plot(combined_df['date'], combined_df['quantity'], label='Arrival Quantity', 
                color=self.colors['quantity'], linewidth=3, marker='o')
        
        ax2.set_title('Quantity Trends', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Quantity (Tonnes)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Combined Analysis: {commodity_name} in {location}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Combined analysis saved to {save_path}")
        plt.show()
        
        # Print correlation analysis
        correlation = combined_df['quantity'].corr(combined_df['modal_price'])
        print(f"\n📊 Correlation Analysis for {commodity_name} in {location}:")
        print(f"Price-Quantity Correlation: {correlation:.3f}")
        if abs(correlation) > 0.7:
            print("Strong correlation detected")
        elif abs(correlation) > 0.3:
            print("Moderate correlation detected")
        else:
            print("Weak correlation detected")


def main():
    """Main function to run tests and create visualizations"""
    print("🌾 Agri Sahayak Endpoint Testing and Visualization Tool\n")
    
    # Initialize tester and visualizer
    tester = EndpointTester()
    visualizer = DataVisualizer()
    
    # Run all tests
    test_results = tester.run_all_tests()
    
    if test_results:
        print("🎯 All tests completed! Creating visualizations...\n")
        
        # Create visualizations if we have data
        if test_results.get("prices"):
            print("📈 Creating price analysis...")
            visualizer.plot_prices(
                test_results["prices"],
                commodity_name="Sample Commodity",
                location="Sample Location"
            )
        
        if test_results.get("quantities"):
            print("📊 Creating quantity analysis...")
            visualizer.plot_quantities(
                test_results["quantities"],
                commodity_name="Sample Commodity",
                location="Sample Location"
            )
        
        if test_results.get("prices") and test_results.get("quantities"):
            print("🔗 Creating combined analysis...")
            visualizer.plot_combined_analysis(
                test_results["prices"],
                test_results["quantities"],
                commodity_name="Sample Commodity",
                location="Sample Location"
            )
    
    print("\n✨ Testing and visualization complete!")


if __name__ == "__main__":
    main()
