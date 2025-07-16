#!/usr/bin/env python3
"""
Test script to verify map functionality is working correctly
"""

import pandas as pd
import sys
sys.path.append('.')

# Import the updated functions
from catastro_property_search import (
    get_geometry_for_results, 
    extract_geometry_center, 
    convert_coordinates,
    create_map
)

def test_map_functionality():
    print("🗺️ Testing Map Functionality")
    print("="*50)
    
    # Create mock search results
    search_results = pd.DataFrame({
        'referencia_catastral': ['000201400DD89C', '000300100DD89C', '000300500DD89C'],
        'municipio': ['ALAIOR', 'ALAIOR', 'ALAIOR'],
        'superficie_parcela': [1000, 1500, 2000],
        'total_built_area': [500, 750, 1000],
        'num_buildings': [2, 3, 4],
        'num_units': [4, 6, 8],
        'match_score': [85.5, 90.2, 78.3]
    })
    
    print(f"📊 Mock search results: {len(search_results)} properties")
    
    # Test 1: Geometry loading
    print("\n🔍 Test 1: Geometry Loading")
    try:
        geometry_df = get_geometry_for_results(search_results)
        if geometry_df is not None and not geometry_df.empty:
            print(f"✅ Geometry loaded: {len(geometry_df)} parcels")
            print(f"   Columns: {list(geometry_df.columns)}")
            
            # Show sample data
            for _, row in geometry_df.head(2).iterrows():
                print(f"   {row['referencia_catastral']}: X={row['center_x']:.2f}, Y={row['center_y']:.2f}")
        else:
            print("❌ No geometry data loaded")
            return False
    except Exception as e:
        print(f"❌ Geometry loading failed: {e}")
        return False
    
    # Test 2: Coordinate conversion
    print("\n🌍 Test 2: Coordinate Conversion")
    try:
        test_coords = [
            (geometry_df.iloc[0]['center_x'], geometry_df.iloc[0]['center_y']),
            (geometry_df.iloc[1]['center_x'], geometry_df.iloc[1]['center_y'])
        ]
        
        for i, (x, y) in enumerate(test_coords):
            lat, lon = extract_geometry_center(x, y)
            if lat is not None and lon is not None:
                print(f"✅ Conversion {i+1}: UTM({x:.2f}, {y:.2f}) -> LatLon({lat:.6f}, {lon:.6f})")
                # Check if coordinates are reasonable for Spain
                if 35.0 <= lat <= 45.0 and -12.0 <= lon <= 6.0:
                    print(f"   ✅ Valid Spanish coordinates")
                else:
                    print(f"   ❌ Invalid coordinates for Spain")
            else:
                print(f"❌ Conversion {i+1}: Failed")
                return False
    except Exception as e:
        print(f"❌ Coordinate conversion failed: {e}")
        return False
    
    # Test 3: Map creation
    print("\n🗺️ Test 3: Map Creation")
    try:
        # Check if folium is available
        try:
            import folium
            print("✅ Folium library available")
        except ImportError:
            print("❌ Folium library not available - map won't work")
            return False
        
        # Create map
        map_obj = create_map(search_results, geometry_df)
        if map_obj is not None:
            print("✅ Map created successfully")
            if hasattr(map_obj, 'status_info'):
                status = map_obj.status_info
                print(f"   Added: {status['added_count']} markers")
                print(f"   Failed: {status['failed_count']} markers")
                
                if status['added_count'] > 0:
                    print("✅ Map has markers - should display correctly!")
                    return True
                else:
                    print("❌ No markers added to map")
                    return False
            else:
                print("✅ Map created but no status info available")
                return True
        else:
            print("❌ Map creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Map creation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_map_functionality()
    
    print("\n" + "="*50)
    if success:
        print("🎉 MAP FUNCTIONALITY: ALL TESTS PASSED!")
        print("✅ The map should now work correctly in Streamlit")
    else:
        print("❌ MAP FUNCTIONALITY: TESTS FAILED!")
        print("🔧 Please check the issues above") 