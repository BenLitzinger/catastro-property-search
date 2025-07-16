import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple

# Map functionality
try:
    import folium
    from streamlit_folium import st_folium
    MAP_AVAILABLE = True
except ImportError as e:
    MAP_AVAILABLE = False
    # Debug: uncomment to see import error
    # st.error(f"Map import error: {e}")

# Page config
st.set_page_config(
    page_title="Catastro Property Search",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_database_connection():
    """Get database connection for direct queries"""
    try:
        from database_service import catastro_db
        return catastro_db
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        return None

def build_search_query(filters: Dict[str, Any]) -> Tuple[str, List]:
    """Build SQL query based on search filters"""
    
    # Base query with aggregated data (without geometry for GROUP BY)
    base_query = """
    WITH ParcelData AS (
        SELECT 
            p.referencia_catastral,
            p.municipio,
            p.superficie_parcela,
            
            -- Building aggregations
            COUNT(DISTINCT b.id) as num_buildings,
            COALESCE(SUM(b.built_area), 0) as total_built_area,
            (
                SELECT TOP 10 STRING_AGG(CAST(CAST(b2.built_area AS VARCHAR(20)) AS NVARCHAR(MAX)), ',')
                FROM catastro_buildings b2 
                WHERE b2.parcel_ref = p.referencia_catastral
            ) as buildings_areas,
            
            -- Unit aggregations  
            COUNT(DISTINCT u.id) as num_units,
            (
                SELECT TOP 10 STRING_AGG(CAST(LEFT(ISNULL(u2.use_type, ''), 50) AS NVARCHAR(MAX)), ',')
                FROM catastro_units u2 
                WHERE u2.parcel_ref = p.referencia_catastral
            ) as units_use_types,
            (
                SELECT TOP 10 STRING_AGG(CAST(CAST(u2.floor_area AS VARCHAR(20)) AS NVARCHAR(MAX)), ',')
                FROM catastro_units u2 
                WHERE u2.parcel_ref = p.referencia_catastral
            ) as units_floor_areas,
            (
                SELECT TOP 10 STRING_AGG(CAST(CAST(u2.year_built AS VARCHAR(20)) AS NVARCHAR(MAX)), ',')
                FROM catastro_units u2 
                WHERE u2.parcel_ref = p.referencia_catastral
            ) as buildings_years,
            
            -- Calculated metrics
            CASE 
                WHEN p.superficie_parcela > 0 
                THEN (COALESCE(SUM(b.built_area), 0) / p.superficie_parcela) * 100
                ELSE 0 
            END as utilization_score,
            
            CASE 
                WHEN COUNT(DISTINCT b.id) > 0 
                THEN COALESCE(SUM(b.built_area), 0) / COUNT(DISTINCT b.id)
                ELSE 0 
            END as avg_building_area,
            
            CASE 
                WHEN COUNT(DISTINCT u.id) > 0 
                THEN COALESCE(SUM(u.floor_area), 0) / COUNT(DISTINCT u.id)
                ELSE 0 
            END as avg_unit_area
            
        FROM catastro_parcels p
        LEFT JOIN catastro_buildings b ON p.referencia_catastral = b.parcel_ref
        LEFT JOIN catastro_units u ON p.referencia_catastral = u.parcel_ref
        GROUP BY p.referencia_catastral, p.municipio, p.superficie_parcela
    )
    SELECT * FROM ParcelData
    WHERE 1=1
    """
    
    conditions = []
    params = []
    
    # Region filter
    if filters.get('region') and filters['region'] != 'All':
        conditions.append("municipio = ?")
        params.append(filters['region'])
    
    # Parcel area filter
    parcel_range = filters.get('parcel_area_range', (0, 1000000))
    if parcel_range[0] > 0:
        conditions.append("superficie_parcela >= ?")
        params.append(parcel_range[0])
    if parcel_range[1] < 1000000:
        conditions.append("superficie_parcela <= ?")
        params.append(parcel_range[1])
    
    # Built area filter
    built_range = filters.get('built_area_range', (0, 100000))
    area_search_type = filters.get('area_search_type', 'Total built area on parcel')
    
    if built_range[0] > 0 or built_range[1] < 100000:
        if area_search_type == "Total built area on parcel":
            if built_range[0] > 0:
                conditions.append("total_built_area >= ?")
                params.append(built_range[0])
            if built_range[1] < 100000:
                conditions.append("total_built_area <= ?")
                params.append(built_range[1])
        else:
            # Individual building/unit area filtering - more complex
            # We'll handle this in post-processing for now
            pass
    
    # Building count filter
    building_range = filters.get('building_count_range', (0, 999))
    if building_range[0] > 0:
        conditions.append("num_buildings >= ?")
        params.append(building_range[0])
    if building_range[1] < 999:
        conditions.append("num_buildings <= ?")
        params.append(building_range[1])
    
    # Year range filter - we'll handle this in post-processing
    
    # Usage type filter - we'll handle this in post-processing
    
    # Add conditions to query
    if conditions:
        base_query += " AND " + " AND ".join(conditions)
    
    # Add ordering and limit - fetch more rows for better filtering
    max_results = filters.get('max_results', 100)
    # Fetch 10x more rows to ensure we get enough results after filtering
    fetch_limit = max(max_results * 10, 500)  # Minimum 500 rows
    base_query += f" ORDER BY total_built_area DESC OFFSET 0 ROWS FETCH NEXT {fetch_limit} ROWS ONLY"
    
    return base_query, params

def execute_search_query(filters: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Execute the search query and return results"""
    
    db = get_database_connection()
    if not db:
        return None
    
    try:
        # Build query
        query, params = build_search_query(filters)
        
        # Execute query
        with st.spinner("🔍 Searching database..."):
            df = db.execute_query(query, params)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Apply additional filters that couldn't be done in SQL
        df = apply_post_processing_filters(df, filters)
        
        # Calculate match scores
        df['match_score'] = df.apply(lambda row: calculate_match_score(row, filters), axis=1)
        
        # Filter by minimum match score
        min_score = filters.get('min_match_score', 0)
        df = df[df['match_score'] >= min_score]
        
        # Sort by match score
        df = df.sort_values('match_score', ascending=False)
        
        # Limit final results
        max_results = filters.get('max_results', 20)
        df = df.head(max_results)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Search query failed: {e}")
        return None

def apply_post_processing_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters that couldn't be done in SQL"""
    
    if df.empty:
        return df
    
    # Individual building/unit area filtering
    area_search_type = filters.get('area_search_type', 'Total built area on parcel')
    built_range = filters.get('built_area_range', (0, 100000))
    
    if area_search_type == "Individual building/unit area" and (built_range[0] > 0 or built_range[1] < 100000):
        def has_individual_area_in_range(row):
            # Check building areas
            building_areas = parse_structured_data(row.get('buildings_areas', ''), 'csv')
            for area_str in building_areas:
                try:
                    area = float(area_str)
                    if built_range[0] <= area <= built_range[1]:
                        return True
                except (ValueError, TypeError):
                    continue
            
            # Check unit areas
            unit_areas = parse_structured_data(row.get('units_floor_areas', ''), 'csv')
            for area_str in unit_areas:
                try:
                    area = float(area_str)
                    if built_range[0] <= area <= built_range[1]:
                        return True
                except (ValueError, TypeError):
                    continue
            
            return False
        
        df = df[df.apply(has_individual_area_in_range, axis=1)]
    
    # Year range filtering
    year_range = filters.get('year_range', (1900, datetime.now().year))
    if year_range[0] > 1900 or year_range[1] < datetime.now().year:
        def has_year_in_range(row):
            years = parse_structured_data(row.get('buildings_years', ''), 'csv')
            for year_str in years:
                try:
                    year = int(year_str)
                    if year_range[0] <= year <= year_range[1]:
                        return True
                except (ValueError, TypeError):
                    continue
            return False
        
        df = df[df.apply(has_year_in_range, axis=1)]
    
    # Usage type filtering
    usage_types = filters.get('usage_types', [])
    if usage_types:
        def has_usage_type(row):
            unit_types = parse_structured_data(row.get('units_use_types', ''), 'csv')
            return any(usage_type in unit_types for usage_type in usage_types)
        
        df = df[df.apply(has_usage_type, axis=1)]
    
    return df

def parse_structured_data(data_str: str, format_type: str = 'csv') -> List[str]:
    """Parse structured data string into list"""
    if not data_str or pd.isna(data_str):
        return []
    
    try:
        if format_type == 'csv':
            return [item.strip() for item in str(data_str).split(',') if item.strip()]
        elif format_type == 'json':
            return json.loads(data_str)
        else:
            return [str(data_str)]
    except:
        return []

def calculate_match_score(row: pd.Series, filters: Dict[str, Any]) -> float:
    """Calculate match score based on how well the property matches the search criteria"""
    score = 0
    max_score = 0
    
    # Region match (high weight)
    max_score += 20
    if filters.get('region') == 'All' or row.get('municipio') == filters.get('region'):
        score += 20
    
    # Parcel area match (medium weight)
    max_score += 15
    parcel_range = filters.get('parcel_area_range', (0, 1000000))
    parcel_area = row.get('superficie_parcela', 0)
    if parcel_range[0] <= parcel_area <= parcel_range[1]:
        score += 15
    
    # Built area match (high weight)
    max_score += 25
    built_range = filters.get('built_area_range', (0, 100000))
    built_area = row.get('total_built_area', 0)
    if built_range[0] <= built_area <= built_range[1]:
        score += 25
    
    # Building count match (medium weight)
    max_score += 15
    building_range = filters.get('building_count_range', (0, 999))
    building_count = row.get('num_buildings', 0)
    if building_range[0] <= building_count <= building_range[1]:
        score += 15
    
    # Usage type match (medium weight)
    max_score += 15
    usage_types = filters.get('usage_types', [])
    if not usage_types:
        score += 15  # No filter means all match
    else:
        unit_types = parse_structured_data(row.get('units_use_types', ''), 'csv')
        if any(usage_type in unit_types for usage_type in usage_types):
            score += 15
    
    # Year range match (low weight)
    max_score += 10
    year_range = filters.get('year_range', (1900, datetime.now().year))
    years = parse_structured_data(row.get('buildings_years', ''), 'csv')
    if any(year_range[0] <= int(year) <= year_range[1] for year in years if year.isdigit()):
        score += 10
    
    return (score / max_score) * 100 if max_score > 0 else 0

def get_available_municipalities() -> List[str]:
    """Get list of available municipalities from database"""
    db = get_database_connection()
    if not db:
        return ['All']
    
    try:
        query = "SELECT DISTINCT municipio FROM catastro_parcels WHERE municipio IS NOT NULL ORDER BY municipio"
        df = db.execute_query(query)
        if df is not None and not df.empty:
            return ['All'] + df['municipio'].tolist()
        else:
            return ['All']
    except Exception as e:
        st.error(f"❌ Error loading municipalities: {e}")
        return ['All']

def get_geometry_for_results(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Get geometry data for the search results"""
    if df.empty:
        return None
    
    db = get_database_connection()
    if not db:
        return None
    
    try:
        # Get geometry for the result parcels
        parcel_refs = df['referencia_catastral'].tolist()
        if not parcel_refs:
            return None
        
        # Create placeholders for the query
        placeholders = ','.join(['?' for _ in parcel_refs])
        query = f"""
        SELECT 
            referencia_catastral,
            geometry.STCentroid().STX as center_x,
            geometry.STCentroid().STY as center_y
        FROM catastro_parcels 
        WHERE referencia_catastral IN ({placeholders}) 
        AND geometry IS NOT NULL
        """
        
        geometry_df = db.execute_query(query, parcel_refs)
        return geometry_df
        
    except Exception as e:
        st.error(f"❌ Error loading geometry: {e}")
        return None

def convert_coordinates(x, y, from_epsg=25830, to_epsg=4326):
    """Convert coordinates from one projection to another (requires pyproj)"""
    try:
        import pyproj
        
        # Try newer pyproj API first
        try:
            from pyproj import Transformer
            transformer = Transformer.from_epsg(from_epsg, to_epsg, always_xy=True)
            lon, lat = transformer.transform(x, y)
            return lat, lon
        except AttributeError:
            # Fallback to older pyproj API
            try:
                from pyproj import Proj, transform
                source = Proj(proj='utm', zone=30, ellps='WGS84', datum='WGS84')
                target = Proj(proj='latlong', datum='WGS84')
                lon, lat = transform(source, target, x, y)
                return lat, lon
            except:
                # If both fail, use manual conversion
                raise ImportError("pyproj API incompatible")
                
    except ImportError:
        # Fallback: rough approximation for EPSG:25830 to WGS84 for Spain
        # EPSG:25830 is UTM Zone 30N, so we can do a reasonable approximation
        # Convert from UTM Zone 30N to lat/lon
        
        # UTM Zone 30N parameters
        lat_rad = (y - 4000000) / 6378137.0  # Rough latitude conversion
        lat = lat_rad * 180 / 3.14159 + 36.0  # Convert to degrees, offset for Spain
        
        # Longitude calculation for UTM Zone 30N (central meridian -3°)
        lon_rad = (x - 500000) / (6378137.0 * 0.9996)  # UTM scale factor 0.9996
        lon = lon_rad * 180 / 3.14159 - 3.0  # Central meridian of Zone 30N
        
        return lat, lon
    except Exception as e:
        # Debug: uncomment to see conversion errors
        # st.write(f"Coordinate conversion error: {e}")
        return None, None

def extract_geometry_center(center_x, center_y):
    """Extract center coordinates from SQL Server geometry center coordinates"""
    if center_x is None or center_y is None:
        return None, None
    
    try:
        # Convert to float if they're not already
        x = float(center_x)
        y = float(center_y)
        
        # Convert from UTM to lat/lon
        return convert_coordinates(x, y)
        
    except Exception as e:
        # Debug: uncomment to see geometry errors
        # st.write(f"Geometry extraction error: {e}")
        pass
    
    return None, None

def create_map(results_df: pd.DataFrame, geometry_df: Optional[pd.DataFrame] = None) -> Optional[folium.Map]:
    """Create a Folium map with property markers"""
    if results_df.empty:
        return None
    
    # Get geometry data if not provided
    if geometry_df is None:
        geometry_df = get_geometry_for_results(results_df)
    
    if geometry_df is None or geometry_df.empty:
        st.warning("📍 No geometry data available for mapping")
        return None
    
    # Calculate center based on geometry data
    center_lat, center_lon = 39.5696, 2.6502  # Default to Palma, Mallorca
    zoom_start = 10
    
    # Try to calculate center from search results
    valid_coords = []
    for _, geometry_row in geometry_df.iterrows():
        center_x = geometry_row.get('center_x')
        center_y = geometry_row.get('center_y')
        
        if center_x is not None and center_y is not None:
            lat, lon = extract_geometry_center(center_x, center_y)
            if lat is not None and lon is not None:
                valid_coords.append((lat, lon))
    
    # If we have valid coordinates, center the map on them
    if valid_coords:
        # Calculate the center of all properties
        avg_lat = sum(coord[0] for coord in valid_coords) / len(valid_coords)
        avg_lon = sum(coord[1] for coord in valid_coords) / len(valid_coords)
        center_lat, center_lon = avg_lat, avg_lon
        
        # Adjust zoom based on spread of results
        if len(valid_coords) > 1:
            lat_range = max(coord[0] for coord in valid_coords) - min(coord[0] for coord in valid_coords)
            lon_range = max(coord[1] for coord in valid_coords) - min(coord[1] for coord in valid_coords)
            max_range = max(lat_range, lon_range)
            
            # Adjust zoom based on spread
            if max_range > 0.5:
                zoom_start = 8
            elif max_range > 0.1:
                zoom_start = 10
            elif max_range > 0.05:
                zoom_start = 12
            else:
                zoom_start = 14
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    added_count = 0
    failed_count = 0
    
    # Add markers for each property
    for _, result_row in results_df.iterrows():
        ref_catastral = result_row['referencia_catastral']
        
        # Find geometry for this parcel
        geometry_row = geometry_df[geometry_df['referencia_catastral'] == ref_catastral]
        if geometry_row.empty:
            failed_count += 1
            continue
        
        # Get center coordinates from SQL Server geometry
        center_x = geometry_row.iloc[0]['center_x']
        center_y = geometry_row.iloc[0]['center_y']
        
        # Extract coordinates
        lat, lon = extract_geometry_center(center_x, center_y)
        
        if lat is None or lon is None:
            failed_count += 1
            continue
        
        # Create popup content
        popup_content = f"""
        <b>Property {ref_catastral}</b><br>
        📍 Municipality: {result_row.get('municipio', 'N/A')}<br>
        📏 Parcel Area: {result_row.get('superficie_parcela', 0):,.0f} m²<br>
        🏗️ Built Area: {result_row.get('total_built_area', 0):,.0f} m²<br>
        🏢 Buildings: {result_row.get('num_buildings', 0)}<br>
        🏠 Units: {result_row.get('num_units', 0)}<br>
        🎯 Match Score: {result_row.get('match_score', 0):.1f}%
        """
        
        # Add marker
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"Property {ref_catastral}",
            icon=folium.Icon(color='blue', icon='home')
        ).add_to(m)
        
        added_count += 1
    
    # Store status info on map object
    m.status_info = {
        'added_count': added_count,
        'failed_count': failed_count
    }
    
    return m

def display_property_details(row: pd.Series):
    """Display detailed information about a property"""
    with st.expander(f"📋 Property Details - {row['referencia_catastral']}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information:**")
            st.write(f"• Municipality: {row.get('municipio', 'N/A')}")
            st.write(f"• Parcel Area: {row.get('superficie_parcela', 0):,.0f} m²")
            st.write(f"• Total Built Area: {row.get('total_built_area', 0):,.0f} m²")
            st.write(f"• Utilization Score: {row.get('utilization_score', 0):.1f}%")
            st.write(f"• Match Score: {row.get('match_score', 0):.1f}%")
        
        with col2:
            st.write("**Buildings & Units:**")
            st.write(f"• Number of Buildings: {row.get('num_buildings', 0)}")
            st.write(f"• Number of Units: {row.get('num_units', 0)}")
            st.write(f"• Average Building Area: {row.get('avg_building_area', 0):,.0f} m²")
            st.write(f"• Average Unit Area: {row.get('avg_unit_area', 0):,.0f} m²")
        
        # Show building details
        building_areas = parse_structured_data(row.get('buildings_areas', ''), 'csv')
        building_years = parse_structured_data(row.get('buildings_years', ''), 'csv')
        if building_areas:
            st.write("**Building Details:**")
            building_data = []
            for i, area in enumerate(building_areas):
                year = building_years[i] if i < len(building_years) else 'N/A'
                building_data.append({
                    'Building': i + 1,
                    'Area (m²)': area,
                    'Year Built': year
                })
            if building_data:
                st.dataframe(pd.DataFrame(building_data), use_container_width=True)
        
        # Show unit details
        unit_areas = parse_structured_data(row.get('units_floor_areas', ''), 'csv')
        unit_types = parse_structured_data(row.get('units_use_types', ''), 'csv')
        if unit_areas:
            st.write("**Unit Details:**")
            unit_data = []
            for i, area in enumerate(unit_areas):
                use_type = unit_types[i] if i < len(unit_types) else 'N/A'
                unit_data.append({
                    'Unit': i + 1,
                    'Area (m²)': area,
                    'Use Type': use_type
                })
            if unit_data:
                st.dataframe(pd.DataFrame(unit_data), use_container_width=True)
        
        # Link to official cadastral website
        st.write("**Official Cadastral Information:**")
        cadastral_ref = row.get('referencia_catastral', '')
        if cadastral_ref:
            cadastral_url = f"https://www1.sedecatastro.gob.es/CYCBienInmueble/OVCBusqueda.aspx?fromVolver=ListaBienes&tipoVia=&via=&num=&blq=&esc=&plt=&pta=&descProv=&prov=&mun=&descMuni=&TipUR=&codVia=&comVia=&final=&pest=rc&pol=&par=&Idufir=&RCCompleta={cadastral_ref}&latitud=&longitud=&gradoslat=&minlat=&seglat=&gradoslon=&minlon=&seglon=&x=&y=&huso=&tipoCoordenadas="
            st.markdown(f"🔗 **[View on Official Spanish Cadastral Website]({cadastral_url})**")
            st.info("📋 This link opens the official Spanish cadastral database with detailed property information including legal descriptions, ownership details, and official measurements.")
        else:
            st.warning("⚠️ No cadastral reference available for this property")

def map_new_categories_to_cadastral_types(new_categories: List[str]) -> List[str]:
    """Map new platform categories to original Spanish cadastral usage types"""
    
    # Mapping from new categories to original Spanish cadastral types
    category_mapping = {
        'House': ['Residencial'],
        'Apartment': ['Residencial'],
        'Land': [
            'Agrario', 
            'Almacen agrario', 
            'Industrial agrario', 
            'Obras de urbanización y jardineria, suelos sin edificar'
        ],
        'Commercial': [
            'Comercial', 
            'Industrial', 
            'Oficinas', 
            'Ocio y Hosteleria', 
            'Almacen-Estacionamiento', 
            'Almacen-Estacionamiento.Uso Industrial'
        ],
        'Special': [
            'Cultural', 
            'Deportivo', 
            'Espectaculos', 
            'Religioso', 
            'Sanidad y Beneficencia', 
            'RDL 1/2004 8.2a', 
            'RDL 1/2004 8.2d', 
            'Almacen-Estacionamiento.Uso Residencial'
        ]
    }
    
    # Convert selected new categories to original cadastral types
    mapped_types = []
    for category in new_categories:
        if category in category_mapping:
            mapped_types.extend(category_mapping[category])
    
    return mapped_types

def main():
    st.title("🏠 Catastro Property Search System")
    st.markdown("Search and explore Spanish cadastral properties with advanced filtering and ranking")
    
    # Database connection status
    with st.sidebar.expander("🔗 Database Status", expanded=False):
        db = get_database_connection()
        if db and db.test_connection():
            st.success("✅ Database connection active")
        else:
            st.error("❌ Database connection failed")
    
    # Sidebar filters
    st.sidebar.header("🔍 Search Filters")
    
    # Region filter
    with st.spinner("Loading municipalities..."):
        regions = get_available_municipalities()
    selected_region = st.sidebar.selectbox("📍 Region (Municipality)", regions)
    
    # Parcel area filter
    st.sidebar.subheader("📏 Parcel Area")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        parcel_min_exact = st.number_input(
            "Min (m²)", 
            min_value=0, 
            max_value=1000000, 
            value=0,
            step=100,
            key="parcel_min"
        )
    with col2:
        parcel_max_exact = st.number_input(
            "Max (m²)", 
            min_value=0, 
            max_value=1000000, 
            value=100000,
            step=100,
            key="parcel_max"
        )
    
    final_parcel_range = (parcel_min_exact, parcel_max_exact)
    
    # Built area filter
    st.sidebar.subheader("🏗️ Built Area")
    area_search_type = st.sidebar.radio(
        "Search Type:",
        ["Total built area on parcel", "Individual building/unit area"],
        help="Choose whether to search by total area or individual building/unit size"
    )
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        built_min_exact = st.number_input(
            "Min (m²)", 
            min_value=0, 
            max_value=100000, 
            value=0,
            step=10,
            key="built_min"
        )
    with col4:
        built_max_exact = st.number_input(
            "Max (m²)", 
            min_value=0, 
            max_value=100000, 
            value=1000,
            step=10,
            key="built_max"
        )
    
    final_built_range = (built_min_exact, built_max_exact)
    
    # Year range filter
    current_year = datetime.now().year
    year_range = st.sidebar.slider(
        "📅 Year Built Range", 
        1900, current_year, 
        (1970, current_year),
        step=5
    )
    
    # Usage type filter
    st.sidebar.subheader("🏘️ Usage Types")
    
    # New platform categories
    new_usage_categories = [
        'House', 'Apartment', 'Land', 'Commercial', 'Special'
    ]
    
    selected_new_categories = st.sidebar.multiselect(
        "Select Usage Types",
        new_usage_categories,
        default=[],
        help="Select one or more usage types to filter properties"
    )
    
    # Convert new categories to original Spanish cadastral types
    selected_usage_types = map_new_categories_to_cadastral_types(selected_new_categories)
    
    # Building count filter
    building_count_range = st.sidebar.slider(
        "🏢 Number of Buildings", 
        0, 50, 
        (0, 50),
        step=1
    )
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        max_results = st.number_input("📊 Maximum Results", 1, 100, 20)
        min_match_score = st.slider("🎯 Minimum Match Score (%)", 0, 100, 0)
    
    # Search button
    if st.sidebar.button("🔍 Search Properties", type="primary"):
        
        # Compile filters
        filters = {
            'region': selected_region,
            'parcel_area_range': final_parcel_range,
            'built_area_range': final_built_range,
            'area_search_type': area_search_type,
            'year_range': year_range,
            'usage_types': selected_usage_types,
            'building_count_range': building_count_range,
            'max_results': max_results,
            'min_match_score': min_match_score
        }
        
        # Execute search
        results_df = execute_search_query(filters)
        
        if results_df is not None:
            st.session_state['search_results'] = results_df
            st.session_state['search_performed'] = True
            
            # Show immediate feedback
            if not results_df.empty:
                st.success(f"🎯 Found {len(results_df)} matching properties!")
            else:
                st.warning("🔍 No properties found matching your criteria. Try adjusting your filters.")
        else:
            st.error("❌ Search failed. Please try again.")
    
    # Display results
    if st.session_state.get('search_performed', False):
        results_df = st.session_state.get('search_results', pd.DataFrame())
        
        if not results_df.empty:
            st.header(f"🎯 Search Results ({len(results_df)} properties found)")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Match Score", f"{results_df['match_score'].mean():.1f}%")
            with col2:
                st.metric("Best Match Score", f"{results_df['match_score'].max():.1f}%")
            with col3:
                st.metric("Total Parcel Area", f"{results_df['superficie_parcela'].sum():,.0f} m²")
            with col4:
                st.metric("Total Built Area", f"{results_df['total_built_area'].sum():,.0f} m²")
            
            # Map display
            if MAP_AVAILABLE:
                st.subheader("🗺️ Property Locations")
                
                # Option to show/hide map
                show_map = st.checkbox("Show properties on map", value=True)
                
                if show_map:
                    # Limit properties for map display (performance)
                    map_limit = min(50, len(results_df))
                    if len(results_df) > 50:
                        st.info(f"📍 Showing first {map_limit} properties on map (of {len(results_df)} total)")
                    
                    map_df = results_df.head(map_limit)
                    
                    # Create and display map
                    map_obj = create_map(map_df)
                    if map_obj:
                        # Display compact status message before the map
                        if hasattr(map_obj, 'status_info'):
                            status = map_obj.status_info
                            if status['added_count'] > 0:
                                if status['failed_count'] > 0:
                                    st.info(f"📍 Showing {status['added_count']} properties on map ({status['failed_count']} could not be plotted)")
                                else:
                                    st.info(f"📍 Showing {status['added_count']} properties on map")
                            else:
                                st.warning("📍 No properties could be plotted on map")
                        
                        # Display the map
                        st_folium(map_obj, width=700, height=400)
                    else:
                        st.warning("📍 Unable to create map")
            else:
                st.info("📍 Install folium and streamlit-folium to enable map functionality")
            
            # Results table
            st.subheader("📊 Property Results")
            
            # Display options
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox(
                    "Sort by:", 
                    ["Match Score", "Parcel Area", "Built Area", "Utilization Score", "Municipality"],
                    index=0
                )
            with col2:
                sort_ascending = st.checkbox("Sort Ascending", value=False)
            
            # Apply sorting
            sort_columns = {
                "Match Score": "match_score",
                "Parcel Area": "superficie_parcela", 
                "Built Area": "total_built_area",
                "Utilization Score": "utilization_score",
                "Municipality": "municipio"
            }
            
            if sort_by in sort_columns:
                results_df = results_df.sort_values(sort_columns[sort_by], ascending=sort_ascending)
            
            # Display results
            for idx, row in results_df.iterrows():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    st.write(f"**{row['referencia_catastral']}**")
                    st.write(f"📍 {row.get('municipio', 'N/A')}")
                
                with col2:
                    st.write(f"📏 {row.get('superficie_parcela', 0):,.0f} m²")
                    st.write(f"🏗️ {row.get('total_built_area', 0):,.0f} m²")
                
                with col3:
                    st.write(f"🏢 {row.get('num_buildings', 0)} buildings")
                    st.write(f"🏠 {row.get('num_units', 0)} units")
                
                with col4:
                    st.write(f"🎯 {row.get('match_score', 0):.1f}%")
                    st.write(f"📊 {row.get('utilization_score', 0):.1f}%")
                
                # Property details
                display_property_details(row)
                
                st.divider()
        
        else:
            st.info("🔍 No properties found matching your criteria. Try adjusting your search filters.")
    
    else:
        # Show welcome message
        st.info("👋 Welcome! Use the filters in the sidebar and click 'Search Properties' to find cadastral properties.")
        
        # Show some basic stats
        with st.expander("📊 Database Overview", expanded=True):
            db = get_database_connection()
            if db:
                try:
                    # Get basic stats
                    stats_query = """
                    SELECT 
                        COUNT(*) as total_parcels,
                        COUNT(DISTINCT municipio) as total_municipalities,
                        AVG(superficie_parcela) as avg_parcel_area,
                        SUM(superficie_parcela) as total_parcel_area
                    FROM catastro_parcels
                    """
                    stats_df = db.execute_query(stats_query)
                    
                    if stats_df is not None and not stats_df.empty:
                        stats = stats_df.iloc[0]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Parcels", f"{stats['total_parcels']:,}")
                        with col2:
                            st.metric("Municipalities", f"{stats['total_municipalities']:,}")
                        with col3:
                            st.metric("Avg Parcel Area", f"{stats['avg_parcel_area']:,.0f} m²")
                        with col4:
                            st.metric("Total Land Area", f"{stats['total_parcel_area']:,.0f} m²")
                
                except Exception as e:
                    st.error(f"❌ Error loading database stats: {e}")

if __name__ == "__main__":
    main() 