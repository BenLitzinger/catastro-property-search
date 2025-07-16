#!/usr/bin/env python3
"""
Database Service Layer for Catastro Property Search
Replaces static Excel file loading with dynamic database queries
"""

import pyodbc
import pandas as pd
import json
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CatastroDatabase:
    """Database connection and data processing service for Catastro data"""
    
    def __init__(self):
        # Database connection parameters
        self.server = "hellodata-database.database.windows.net"
        self.database = "hellodata"
        self.username = "hellodata_prod"
        self.password = "Dvd^1i83]70q"
        self.driver = self._get_available_driver()
        
        # Table names
        self.PARCEL_TABLE = "catastro_parcels"
        self.UNIT_TABLE = "catastro_units"
        self.BUILDING_TABLE = "catastro_buildings"
        
        # Connection string
        self.conn_str = f"DRIVER={self.driver};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"
        
        logger.info(f"Database service initialized with driver: {self.driver}")
    
    def _get_available_driver(self) -> str:
        """Get the best available ODBC driver"""
        drivers = pyodbc.drivers()
        
        # Preferred drivers in order
        preferred_drivers = [
            "ODBC Driver 18 for SQL Server",
            "ODBC Driver 17 for SQL Server", 
            "ODBC Driver 13 for SQL Server",
            "SQL Server"
        ]
        
        for driver in preferred_drivers:
            if driver in drivers:
                return driver
        
        raise RuntimeError("No compatible ODBC driver found")
    
    def get_connection(self) -> pyodbc.Connection:
        """Get database connection with error handling"""
        try:
            return pyodbc.connect(self.conn_str)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        try:
            with self.get_connection() as conn:
                df = pd.read_sql(query, conn, params=params)
                return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def load_parcels_data(self) -> pd.DataFrame:
        """Load parcels data from database"""
        logger.info("Loading parcels data from database...")
        
        query = f"""
        SELECT 
            id,
            referencia_catastral,
            municipio,
            codigo_municipio,
            provincia,
            codigo_provincia,
            superficie_parcela,
            uso_parcela,
            geometry,
            last_update
        FROM {self.PARCEL_TABLE}
        """
        
        df = self.execute_query(query)
        logger.info(f"Loaded {len(df)} parcels")
        return df
    
    def load_buildings_data(self) -> pd.DataFrame:
        """Load buildings data from database"""
        logger.info("Loading buildings data from database...")
        
        query = f"""
        SELECT 
            id,
            parcel_ref,
            building_type,
            description,
            built_area,
            staircase,
            floor,
            door,
            municipality,
            province,
            last_update
        FROM {self.BUILDING_TABLE}
        """
        
        df = self.execute_query(query)
        logger.info(f"Loaded {len(df)} buildings")
        return df
    
    def load_units_data(self) -> pd.DataFrame:
        """Load units data from database"""
        logger.info("Loading units data from database...")
        
        query = f"""
        SELECT 
            id,
            parcel_ref,
            unit_ref,
            car,
            cc1,
            cc2,
            province_code,
            municipality_code,
            cadastral_municipio,
            use_type,
            floor_area,
            year_built,
            participation,
            street_name,
            street_type,
            street_code,
            portal_number,
            portal_suffix,
            floor,
            door,
            staircase,
            postal_code,
            address_code,
            local_zoning_code,
            polygon_number,
            parcel_number,
            local_place_name,
            parcel_grouping_code,
            province,
            municipality,
            raw_address_text,
            num_entries,
            spr_code,
            spr_type_code,
            spr_type_desc,
            spr_url,
            spr_entity_name,
            last_update,
            geometry
        FROM {self.UNIT_TABLE}
        """
        
        df = self.execute_query(query)
        logger.info(f"Loaded {len(df)} units")
        return df
    
    def aggregate_buildings_data(self, df_buildings: pd.DataFrame) -> pd.DataFrame:
        """Aggregate buildings data by parcel (replicating notebook logic)"""
        logger.info("Aggregating buildings data...")
        
        building_ref_col = 'parcel_ref'
        
        # Create building aggregations
        building_agg = df_buildings.groupby(building_ref_col).agg({
            'id': 'count',  # Count of buildings per parcel
            'built_area': ['sum', 'mean', 'max'],  # Built area statistics
        }).round(2)
        
        # Flatten column names
        building_agg.columns = ['num_buildings', 'total_built_area', 'avg_built_area', 'max_built_area']
        
        # Add individual building details - structured approach
        def process_buildings_for_parcel(group):
            # Lists for CSV format
            built_areas = []
            building_types = []
            floors = []
            descriptions = []
            
            # Detailed dict for JSON format
            building_details = []
            
            for idx, building in group.iterrows():
                built_area = building['built_area'] if pd.notna(building['built_area']) else 0
                building_type = building['building_type'] if pd.notna(building['building_type']) else 'Unknown'
                floor = building['floor'] if pd.notna(building['floor']) else 'Unknown'
                description = building['description'] if pd.notna(building['description']) else 'Unknown'
                staircase = building['staircase'] if pd.notna(building['staircase']) else 'Unknown'
                door = building['door'] if pd.notna(building['door']) else 'Unknown'
                
                # Add to lists (for CSV columns)
                built_areas.append(built_area)
                building_types.append(building_type)
                floors.append(floor)
                descriptions.append(description)
                
                # Add to detailed structure (for JSON column)
                building_details.append({
                    'building_id': int(building['id']),
                    'built_area': built_area,
                    'building_type': building_type,
                    'description': description,
                    'floor': floor,
                    'staircase': staircase,
                    'door': door
                })
            
            return {
                'areas_csv': ', '.join(map(str, built_areas)),
                'types_csv': ', '.join(building_types),
                'floors_csv': ', '.join(floors),
                'descriptions_csv': ', '.join(descriptions),
                'buildings_details_json': json.dumps(building_details, ensure_ascii=False)
            }
        
        # Process all buildings for each parcel
        buildings_processed = df_buildings.groupby(building_ref_col).apply(process_buildings_for_parcel)
        
        # Add the structured building data to aggregations
        building_agg['buildings_areas'] = buildings_processed.apply(lambda x: x['areas_csv'])
        building_agg['buildings_types'] = buildings_processed.apply(lambda x: x['types_csv'])
        building_agg['buildings_floors'] = buildings_processed.apply(lambda x: x['floors_csv'])
        building_agg['buildings_descriptions'] = buildings_processed.apply(lambda x: x['descriptions_csv'])
        building_agg['buildings_details_json'] = buildings_processed.apply(lambda x: x['buildings_details_json'])
        
        # Add building type diversity
        building_type_counts = df_buildings.groupby(building_ref_col)['building_type'].agg(['nunique', 'count'])
        building_agg['unique_building_types'] = building_type_counts['nunique']
        building_agg['building_type_diversity'] = building_type_counts['nunique'] / building_type_counts['count']
        
        # Most common building type per parcel
        most_common_type = df_buildings.groupby(building_ref_col)['building_type'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown')
        building_agg['primary_building_type'] = most_common_type
        
        building_agg = building_agg.reset_index()
        return building_agg
    
    def aggregate_units_data(self, df_units: pd.DataFrame) -> pd.DataFrame:
        """Aggregate units data by parcel (replicating notebook logic)"""
        logger.info("Aggregating units data...")
        
        unit_ref_col = 'parcel_ref'
        
        # Create unit aggregations
        unit_agg = df_units.groupby(unit_ref_col).agg({
            'id': 'count',  # Count of units per parcel
            'floor_area': ['sum', 'mean', 'max', 'min'],  # Floor area statistics
            'year_built': ['mean', 'min', 'max'],  # Building year statistics
            'participation': ['sum', 'mean'],  # Participation statistics
        }).round(2)
        
        # Flatten column names
        unit_agg.columns = [
            'num_units', 
            'total_floor_area', 'avg_floor_area', 'max_floor_area', 'min_floor_area',
            'avg_year_built', 'oldest_year_built', 'newest_year_built',
            'total_participation', 'avg_participation'
        ]
        
        # Add individual unit details - structured approach
        def process_units_for_parcel(group):
            current_year = datetime.now().year
            
            # Lists for CSV format
            years_built = []
            ages = []
            floor_areas = []
            use_types = []
            
            # Detailed dict for JSON format
            unit_details = []
            
            for idx, unit in group.iterrows():
                year_built = unit['year_built'] if pd.notna(unit['year_built']) else None
                age = current_year - year_built if year_built is not None else None
                floor_area = unit['floor_area'] if pd.notna(unit['floor_area']) else 0
                use_type = unit['use_type'] if pd.notna(unit['use_type']) else 'Unknown'
                
                # Add to lists (for CSV columns)
                if year_built is not None:
                    years_built.append(int(year_built))
                    ages.append(int(age))
                floor_areas.append(floor_area)
                use_types.append(use_type)
                
                # Add to detailed structure (for JSON column)
                unit_details.append({
                    'unit_id': int(unit['id']),
                    'year_built': int(year_built) if year_built is not None else None,
                    'age': int(age) if age is not None else None,
                    'floor_area': floor_area,
                    'use_type': use_type,
                    'participation': unit['participation'] if pd.notna(unit['participation']) else 0
                })
            
            return {
                'years_built_csv': ', '.join(map(str, years_built)),
                'ages_csv': ', '.join(map(str, ages)),
                'floor_areas_csv': ', '.join(map(str, floor_areas)),
                'use_types_csv': ', '.join(use_types),
                'units_details_json': json.dumps(unit_details, ensure_ascii=False)
            }
        
        # Process all units for each parcel
        units_processed = df_units.groupby(unit_ref_col).apply(process_units_for_parcel)
        
        # Add the structured unit data to aggregations
        unit_agg['units_years_built'] = units_processed.apply(lambda x: x['years_built_csv'])
        unit_agg['units_ages'] = units_processed.apply(lambda x: x['ages_csv'])
        unit_agg['units_floor_areas'] = units_processed.apply(lambda x: x['floor_areas_csv'])
        unit_agg['units_use_types'] = units_processed.apply(lambda x: x['use_types_csv'])
        unit_agg['units_details_json'] = units_processed.apply(lambda x: x['units_details_json'])
        
        # Add use type diversity and statistics
        use_type_stats = df_units.groupby(unit_ref_col)['use_type'].agg(['nunique', 'count'])
        unit_agg['unique_use_types'] = use_type_stats['nunique']
        unit_agg['use_type_diversity'] = use_type_stats['nunique'] / use_type_stats['count']
        
        # Most common use type per parcel
        most_common_use = df_units.groupby(unit_ref_col)['use_type'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown')
        unit_agg['primary_use_type'] = most_common_use
        
        # Calculate residential vs non-residential ratios
        def calc_residential_ratio(group):
            total = len(group)
            residential = (group == 'Residencial').sum()
            return residential / total if total > 0 else 0
        
        residential_ratio = df_units.groupby(unit_ref_col)['use_type'].apply(calc_residential_ratio)
        unit_agg['residential_ratio'] = residential_ratio
        
        # Province and municipality (taking first occurrence per parcel)
        location_info = df_units.groupby(unit_ref_col)[['province', 'municipality']].first()
        unit_agg['province'] = location_info['province']
        unit_agg['municipality'] = location_info['municipality']
        
        unit_agg = unit_agg.reset_index()
        return unit_agg
    
    def add_computed_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed metrics (replicating notebook logic)"""
        logger.info("Adding computed metrics...")
        
        # Building density (buildings per parcel area)
        if 'num_buildings' in df.columns and 'superficie_parcela' in df.columns:
            df['building_density_per_sqm'] = df['num_buildings'] / (df['superficie_parcela'] + 1e-6)
        
        # Unit density (units per parcel area)  
        if 'num_units' in df.columns and 'superficie_parcela' in df.columns:
            df['unit_density_per_sqm'] = df['num_units'] / (df['superficie_parcela'] + 1e-6)
        
        # Average units per building
        if 'num_units' in df.columns and 'num_buildings' in df.columns:
            df['avg_units_per_building'] = df['num_units'] / (df['num_buildings'] + 1e-6)
            df['avg_units_per_building'] = df['avg_units_per_building'].replace([np.inf, -np.inf], 0)
        
        # Development intensity (built area vs parcel area)
        if 'total_built_area' in df.columns and 'superficie_parcela' in df.columns:
            df['building_coverage_ratio'] = df['total_built_area'] / (df['superficie_parcela'] + 1e-6)
            df['building_coverage_ratio'] = df['building_coverage_ratio'].replace([np.inf, -np.inf], 0)
        
        # Floor area ratio (total floor area vs parcel area)
        if 'total_floor_area' in df.columns and 'superficie_parcela' in df.columns:
            df['floor_area_ratio'] = df['total_floor_area'] / (df['superficie_parcela'] + 1e-6)
            df['floor_area_ratio'] = df['floor_area_ratio'].replace([np.inf, -np.inf], 0)
        
        # Combined total constructed area (buildings + units)
        if 'total_built_area' in df.columns and 'total_floor_area' in df.columns:
            df['total_constructed_area'] = df['total_built_area'] + df['total_floor_area']
        
        # Development age metrics
        if 'avg_year_built' in df.columns:
            current_year = datetime.now().year
            df['avg_building_age'] = current_year - df['avg_year_built']
            df['avg_building_age'] = df['avg_building_age'].fillna(0)
        
        # Property value indicators (based on area, age, type)
        # Efficiency ratio (floor area per building area)
        if 'total_floor_area' in df.columns and 'total_built_area' in df.columns:
            df['area_efficiency_ratio'] = df['total_floor_area'] / (df['total_built_area'] + 1e-6)
            df['area_efficiency_ratio'] = df['area_efficiency_ratio'].replace([np.inf, -np.inf], 0)
        
        # Parcel utilization score (0-1 scale)
        if all(col in df.columns for col in ['building_coverage_ratio', 'floor_area_ratio', 'num_buildings', 'superficie_parcela']):
            # Normalize factors for scoring
            coverage_norm = np.clip(df['building_coverage_ratio'], 0, 1)
            far_norm = np.clip(df['floor_area_ratio'] / 3, 0, 1)  # FAR of 3 = max score
            density_norm = np.clip(df['building_density_per_sqm'] * 100, 0, 1)  # Normalize density
            
            df['utilization_score'] = (coverage_norm * 0.4 + far_norm * 0.4 + density_norm * 0.2).round(3)
        
        return df
    
    def process_comprehensive_data(self) -> pd.DataFrame:
        """Process and merge all data (replicating notebook logic)"""
        logger.info("Processing comprehensive catastro data...")
        
        # Load base data
        df_parcels = self.load_parcels_data()
        df_buildings = self.load_buildings_data()
        df_units = self.load_units_data()
        
        # Aggregate buildings and units
        building_agg = self.aggregate_buildings_data(df_buildings)
        unit_agg = self.aggregate_units_data(df_units)
        
        # Start with parcels as the base
        final_df = df_parcels.copy()
        
        # Merge buildings aggregations
        final_df = final_df.merge(building_agg, left_on='referencia_catastral', right_on='parcel_ref', how='left')
        
        # Merge units aggregations  
        final_df = final_df.merge(unit_agg, left_on='referencia_catastral', right_on='parcel_ref', how='left')
        
        # Fill NaN values for count and numeric columns with appropriate defaults
        count_columns = [col for col in final_df.columns if col.startswith('num_')]
        area_columns = [col for col in final_df.columns if 'area' in col.lower() and col.startswith(('total_', 'avg_', 'max_', 'min_'))]
        ratio_columns = [col for col in final_df.columns if any(term in col.lower() for term in ['ratio', 'diversity'])]
        
        final_df[count_columns] = final_df[count_columns].fillna(0)
        final_df[area_columns] = final_df[area_columns].fillna(0)  
        final_df[ratio_columns] = final_df[ratio_columns].fillna(0)
        
        # Fill categorical columns with appropriate defaults
        categorical_fill_values = {
            'primary_building_type': 'No Buildings',
            'primary_use_type': 'No Units',
            'unique_building_types': 0,
            'unique_use_types': 0
        }
        
        for col, fill_value in categorical_fill_values.items():
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna(fill_value)
        
        # Add computed metrics
        final_df = self.add_computed_metrics(final_df)
        
        logger.info(f"Final comprehensive data shape: {final_df.shape}")
        return final_df
    
    def get_search_ready_data(self) -> pd.DataFrame:
        """Get data optimized for search functionality (equivalent to Excel Search_Ready_Data sheet)"""
        logger.info("Generating search-ready data...")
        
        # Process comprehensive data
        final_df = self.process_comprehensive_data()
        
        # Select key columns optimized for search functionality
        parcel_ref_col = 'referencia_catastral'
        key_cols = [
            parcel_ref_col, 'superficie_parcela', 'municipio', 'provincia',
            'num_buildings', 'num_units', 
            'total_built_area', 'total_floor_area', 'total_constructed_area',
            'primary_building_type', 'primary_use_type', 'residential_ratio',
            'avg_year_built', 'avg_building_age',
            'building_density_per_sqm', 'unit_density_per_sqm', 'floor_area_ratio',
            'building_coverage_ratio', 'utilization_score',
            # Add structured details to search sheet for easy filtering
            'units_years_built', 'units_ages', 'units_use_types',
            'buildings_areas', 'buildings_types'
        ]
        
        # Filter to available columns
        available_key_cols = [col for col in key_cols if col in final_df.columns]
        search_df = final_df[available_key_cols].copy()
        
        logger.info(f"Search-ready data shape: {search_df.shape}")
        return search_df
    
    def get_geometry_data(self) -> pd.DataFrame:
        """Get geometry data for map functionality"""
        logger.info("Loading geometry data...")
        
        query = f"""
        SELECT 
            referencia_catastral,
            geometry
        FROM {self.PARCEL_TABLE}
        WHERE geometry IS NOT NULL
        """
        
        df = self.execute_query(query)
        logger.info(f"Loaded geometry data for {len(df)} parcels")
        return df
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Global instance
catastro_db = CatastroDatabase() 