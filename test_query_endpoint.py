#!/usr/bin/env python3
"""
Test script for the updated /api/query endpoint with multipart form data
"""

import requests
import json
import base64
from pathlib import Path

def test_query_endpoint():
    """Test the /api/query endpoint with and without image"""
    
    # Base URL
    base_url = "http://localhost:8000"
    
    # Test data
    test_data = {
        "query": "मेरी गेहूं की फसल में बीमारी है, क्या करना चाहिए?",
        "language": "hindi",
        "latitude": 26.4499,
        "longitude": 80.3319,
        "state_id": 8,  # Uttar Pradesh
        "district_id": json.dumps([104])  # Kanpur Nagar as JSON string
    }
    
    # Test 1: Query without image
    print("=== Testing Query Without Image ===")
    try:
        response = requests.post(
            f"{base_url}/api/query",
            data=test_data,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Response received:")
            print(f"Response: {result.get('response', 'No response')[:200]}...")
            print(f"Agents used: {result.get('agents_used', [])}")
            print(f"Mode: {result.get('mode', 'unknown')}")
            print(f"Query EN: {result.get('query_en', 'No English query')}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    
    # Test 2: Query with image (if disease.png exists)
    print("\n=== Testing Query With Image ===")
    image_path = Path("disease.png")
    
    if image_path.exists():
        try:
            with open(image_path, "rb") as f:
                files = {"image": ("disease.png", f, "image/png")}
                
                response = requests.post(
                    f"{base_url}/api/query",
                    data=test_data,
                    files=files,
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Success! Response with image received:")
                    print(f"Response: {result.get('response', 'No response')[:200]}...")
                    print(f"Agents used: {result.get('agents_used', [])}")
                    print(f"Mode: {result.get('mode', 'unknown')}")
                else:
                    print(f"❌ Error: {response.status_code}")
                    print(f"Response: {response.text}")
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
    else:
        print("⚠️  Image file 'disease.png' not found, skipping image test")
        print("Create a test image file named 'disease.png' to test image upload")

if __name__ == "__main__":
    print("Testing Agri-Sahayak Query Endpoint")
    print("Make sure the backend server is running on localhost:8000")
    print("=" * 50)
    
    test_query_endpoint()
