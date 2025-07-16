#!/usr/bin/env python3
"""
Check characteristics of residential properties to understand appropriate filter defaults
"""

import pandas as pd
from database_service import catastro_db

def check_residential_characteristics():
    """Check the characteristics of residential properties"""
    
    print("🏠 Checking residential property characteristics...")
    
    try:
        # Test database connection
        if not catastro_db.test_connection():
            print("❌ Database connection failed!")
            return
        
        print("✅ Database connection successful!")
        
        # Query to get residential property characteristics
        query = """
        WITH ResidentialParcels AS (
            SELECT 
                p.referencia_catastral,
                p.municipio,
                p.superficie_parcela,
                COUNT(DISTINCT b.id) as num_buildings,
                COALESCE(SUM(b.built_area), 0) as total_built_area,
                COUNT(DISTINCT u.id) as num_units
            FROM catastro_parcels p
            LEFT JOIN catastro_buildings b ON p.referencia_catastral = b.parcel_ref
            LEFT JOIN catastro_units u ON p.referencia_catastral = u.parcel_ref
            WHERE EXISTS (
                SELECT 1 FROM catastro_units u2 
                WHERE u2.parcel_ref = p.referencia_catastral 
                AND u2.use_type = 'Residencial'
            )
            GROUP BY p.referencia_catastral, p.municipio, p.superficie_parcela
        )
        SELECT 
            COUNT(*) as total_residential_parcels,
            AVG(superficie_parcela) as avg_parcel_area,
            MIN(superficie_parcela) as min_parcel_area,
            MAX(superficie_parcela) as max_parcel_area,
            AVG(total_built_area) as avg_built_area,
            MIN(total_built_area) as min_built_area,
            MAX(total_built_area) as max_built_area,
            AVG(num_buildings) as avg_num_buildings,
            AVG(num_units) as avg_num_units
        FROM ResidentialParcels
        """
        
        print("\n📊 Querying residential property characteristics...")
        df = catastro_db.execute_query(query)
        
        if df is not None and not df.empty:
            row = df.iloc[0]
            
            print(f"\n📈 RESIDENTIAL PROPERTY CHARACTERISTICS:")
            print("=" * 60)
            print(f"Total residential parcels: {int(row['total_residential_parcels']):,}")
            print(f"\n📏 PARCEL AREA (m²):")
            print(f"  Average: {row['avg_parcel_area']:,.0f} m²")
            print(f"  Minimum: {row['min_parcel_area']:,.0f} m²")
            print(f"  Maximum: {row['max_parcel_area']:,.0f} m²")
            
            print(f"\n🏗️ BUILT AREA (m²):")
            print(f"  Average: {row['avg_built_area']:,.0f} m²")
            print(f"  Minimum: {row['min_built_area']:,.0f} m²")
            print(f"  Maximum: {row['max_built_area']:,.0f} m²")
            
            print(f"\n🏢 BUILDINGS & UNITS:")
            print(f"  Average buildings per parcel: {row['avg_num_buildings']:.1f}")
            print(f"  Average units per parcel: {row['avg_num_units']:.1f}")
            
            print("=" * 60)
            
            # Check how many residential parcels are excluded by current default filters
            exclusion_query = """
            WITH ResidentialParcels AS (
                SELECT 
                    p.referencia_catastral,
                    p.superficie_parcela,
                    COALESCE(SUM(b.built_area), 0) as total_built_area,
                    COUNT(DISTINCT b.id) as num_buildings
                FROM catastro_parcels p
                LEFT JOIN catastro_buildings b ON p.referencia_catastral = b.parcel_ref
                LEFT JOIN catastro_units u ON p.referencia_catastral = u.parcel_ref
                WHERE EXISTS (
                    SELECT 1 FROM catastro_units u2 
                    WHERE u2.parcel_ref = p.referencia_catastral 
                    AND u2.use_type = 'Residencial'
                )
                GROUP BY p.referencia_catastral, p.superficie_parcela
            )
            SELECT 
                COUNT(*) as total_residential,
                SUM(CASE WHEN superficie_parcela <= 100000 THEN 1 ELSE 0 END) as within_parcel_filter,
                SUM(CASE WHEN total_built_area <= 1000 THEN 1 ELSE 0 END) as within_built_area_filter,
                SUM(CASE WHEN num_buildings <= 50 THEN 1 ELSE 0 END) as within_building_count_filter,
                SUM(CASE 
                    WHEN superficie_parcela <= 100000 
                    AND total_built_area <= 1000 
                    AND num_buildings <= 50 
                    THEN 1 ELSE 0 
                END) as within_all_filters
            FROM ResidentialParcels
            """
            
            print(f"\n🔍 IMPACT OF DEFAULT FILTERS:")
            exclusion_df = catastro_db.execute_query(exclusion_query)
            
            if exclusion_df is not None and not exclusion_df.empty:
                row = exclusion_df.iloc[0]
                total = int(row['total_residential'])
                within_parcel = int(row['within_parcel_filter'])
                within_built = int(row['within_built_area_filter'])
                within_buildings = int(row['within_building_count_filter'])
                within_all = int(row['within_all_filters'])
                
                print(f"  Total residential parcels: {total:,}")
                print(f"  Within parcel area filter (≤100,000 m²): {within_parcel:,} ({within_parcel/total*100:.1f}%)")
                print(f"  Within built area filter (≤1,000 m²): {within_built:,} ({within_built/total*100:.1f}%)")
                print(f"  Within building count filter (≤50): {within_buildings:,} ({within_buildings/total*100:.1f}%)")
                print(f"  Within ALL filters: {within_all:,} ({within_all/total*100:.1f}%)")
                
                print(f"\n💡 RECOMMENDATION:")
                if within_all / total < 0.5:
                    print("  ⚠️  Default filters are too restrictive!")
                    print("  🔧 Consider increasing built area limit to 10,000+ m²")
                    print("  🔧 Consider increasing parcel area limit to 1,000,000+ m²")
                else:
                    print("  ✅ Default filters seem reasonable")
        
        else:
            print("❌ No data returned from query!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_residential_characteristics() 