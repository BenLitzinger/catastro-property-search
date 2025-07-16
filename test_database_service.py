#!/usr/bin/env python3
"""
Test script for the database service layer
Verifies that the database service correctly replicates the Excel file functionality
"""

import pandas as pd
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_service():
    """Test the database service functionality"""
    print("=" * 60)
    print("🧪 TESTING DATABASE SERVICE")
    print("=" * 60)
    
    try:
        from database_service import catastro_db
        
        # Test connection
        print("\n1. Testing database connection...")
        if catastro_db.test_connection():
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            return False
        
        # Test loading search-ready data
        print("\n2. Testing search-ready data loading...")
        search_df = catastro_db.get_search_ready_data()
        
        if search_df is not None and not search_df.empty:
            print(f"✅ Search-ready data loaded successfully: {search_df.shape}")
            print(f"   Columns: {len(search_df.columns)}")
            
            # Check key columns
            key_columns = ['referencia_catastral', 'municipio', 'superficie_parcela', 
                         'num_buildings', 'num_units', 'primary_use_type']
            missing_cols = [col for col in key_columns if col not in search_df.columns]
            if missing_cols:
                print(f"⚠️  Missing key columns: {missing_cols}")
            else:
                print("✅ All key columns present")
                
            # Show sample data
            print("\n   Sample data:")
            print(search_df.head(2))
            
        else:
            print("❌ Failed to load search-ready data")
            return False
        
        # Test geometry data loading
        print("\n3. Testing geometry data loading...")
        geometry_df = catastro_db.get_geometry_data()
        
        if geometry_df is not None and not geometry_df.empty:
            print(f"✅ Geometry data loaded successfully: {geometry_df.shape}")
            print(f"   Parcels with geometry: {len(geometry_df)}")
        else:
            print("⚠️  No geometry data available (map features will be limited)")
        
        # Test data structure consistency
        print("\n4. Testing data structure consistency...")
        
        # Check for structured data columns
        structured_cols = ['units_years_built', 'units_ages', 'units_use_types', 
                          'buildings_areas', 'buildings_types']
        
        for col in structured_cols:
            if col in search_df.columns:
                sample_values = search_df[col].dropna().head(1)
                if not sample_values.empty:
                    print(f"✅ {col}: {sample_values.iloc[0]}")
                else:
                    print(f"⚠️  {col}: No sample data")
            else:
                print(f"❌ {col}: Column missing")
        
        # Test data filtering (simulate search functionality)
        print("\n5. Testing data filtering capabilities...")
        
        # Test municipality filtering
        municipalities = search_df['municipio'].unique()
        print(f"✅ Available municipalities: {list(municipalities)}")
        
        # Test numeric filtering
        numeric_cols = ['superficie_parcela', 'total_built_area', 'num_buildings', 'num_units']
        for col in numeric_cols:
            if col in search_df.columns:
                min_val = search_df[col].min()
                max_val = search_df[col].max()
                print(f"✅ {col}: Range {min_val} - {max_val}")
        
        # Test usage type filtering
        if 'primary_use_type' in search_df.columns:
            usage_types = search_df['primary_use_type'].unique()
            print(f"✅ Available usage types: {list(usage_types)[:5]}...")
        
        print("\n6. Testing performance...")
        start_time = datetime.now()
        
        # Simulate a search query
        filtered_df = search_df[
            (search_df['superficie_parcela'] >= 100) & 
            (search_df['superficie_parcela'] <= 10000) &
            (search_df['num_buildings'] > 0)
        ]
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Search query completed in {duration:.2f} seconds")
        print(f"   Results: {len(filtered_df)} parcels")
        
        print("\n" + "=" * 60)
        print("🎉 DATABASE SERVICE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return False

def compare_with_excel():
    """Compare database results with Excel file (if available)"""
    print("\n" + "=" * 60)
    print("📊 COMPARING WITH EXCEL FILE")
    print("=" * 60)
    
    try:
        import glob
        files = glob.glob("catastro_comprehensive_data_*.xlsx")
        
        if not files:
            print("⚠️  No Excel files found for comparison")
            return
        
        latest_file = max(files)
        print(f"Loading Excel file: {latest_file}")
        
        excel_df = pd.read_excel(latest_file, sheet_name='Search_Ready_Data')
        
        from database_service import catastro_db
        db_df = catastro_db.get_search_ready_data()
        
        print(f"Excel data shape: {excel_df.shape}")
        print(f"Database data shape: {db_df.shape}")
        
        # Compare column names
        excel_cols = set(excel_df.columns)
        db_cols = set(db_df.columns)
        
        missing_in_db = excel_cols - db_cols
        extra_in_db = db_cols - excel_cols
        
        if missing_in_db:
            print(f"⚠️  Columns missing in database: {missing_in_db}")
        if extra_in_db:
            print(f"ℹ️  Extra columns in database: {extra_in_db}")
        
        # Compare row counts
        if len(excel_df) == len(db_df):
            print("✅ Row counts match")
        else:
            print(f"⚠️  Row count difference: Excel={len(excel_df)}, DB={len(db_df)}")
        
        print("📊 Comparison completed")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    success = test_database_service()
    
    if success:
        compare_with_excel()
        print("\n✅ All tests passed! The database service is ready for use.")
    else:
        print("\n❌ Tests failed. Please check the database connection and configuration.") 