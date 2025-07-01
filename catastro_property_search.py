import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

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

# Load and cache data
@st.cache_data
def load_data():
    """Load the processed catastro data"""
    try:
        # Try to load the most recent Excel file
        import glob
        files = glob.glob("catastro_comprehensive_data_*.xlsx")
        if files:
            latest_file = max(files)
            df = pd.read_excel(latest_file, sheet_name='Search_Ready_Data')
            return df
        else:
            st.error("❌ No catastro data files found. Please run the processing notebook first.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

@st.cache_data
def load_full_data_with_geometry():
    """Load the full catastro data including geometry from JSON files"""
    try:
        with open('catastro_parcels.json', 'r', encoding='utf-8') as f:
            parcels_data = json.load(f)
        return pd.DataFrame(parcels_data)
    except Exception as e:
        st.error(f"❌ Error loading geometry data: {e}")
        return None

def convert_coordinates(x, y, from_epsg=25830, to_epsg=4326):
    """Convert coordinates from one projection to another (requires pyproj)"""
    try:
        from pyproj import Transformer
        transformer = Transformer.from_epsg(from_epsg, to_epsg, always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lat, lon
    except ImportError:
        # Fallback: rough approximation for EPSG:25830 to WGS84 for Spain
        # This is a simplified approximation and should not be used for precise mapping
        lat = (y - 4000000) / 111320.0 + 36.0  # Rough conversion for latitude
        lon = (x - 500000) / 111320.0 - 3.0      # Simplified longitude conversion
        return lat, lon
    except Exception as e:
        # Debug: uncomment to see conversion errors
        # st.write(f"Coordinate conversion error: {e}")
        return None, None

def extract_geometry_center(geometry_data):
    """Extract center coordinates from geometry data"""
    if not geometry_data or not isinstance(geometry_data, dict):
        return None, None
    
    try:
        # Try different possible structures
        points = None
        if 'points' in geometry_data:
            points = geometry_data['points']
        elif 'coordinates' in geometry_data:
            points = geometry_data['coordinates']
        elif 'geometry' in geometry_data:
            inner_geom = geometry_data['geometry']
            if isinstance(inner_geom, dict):
                if 'points' in inner_geom:
                    points = inner_geom['points']
                elif 'coordinates' in inner_geom:
                    points = inner_geom['coordinates']
        
        if not points or not isinstance(points, list):
            return None, None
        
        # Handle different coordinate formats
        x_coords = []
        y_coords = []
        
        for point in points:
            x, y = None, None
            if isinstance(point, dict):
                # Format: {'x': 123, 'y': 456}
                x = point.get('x') or point.get('X')
                y = point.get('y') or point.get('Y')
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                # Format: [x, y] or (x, y)
                x, y = point[0], point[1]
            
            if x is not None and y is not None:
                try:
                    x_coords.append(float(x))
                    y_coords.append(float(y))
                except (ValueError, TypeError):
                    continue
        
        if x_coords and y_coords:
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            # Convert to lat/lon
            lat, lon = convert_coordinates(center_x, center_y)
            return lat, lon
    except Exception as e:
        # Debug: uncomment to see errors
        # st.write(f"Debug: Error extracting geometry: {e}")
        pass
    
    return None, None

def create_map(properties_df, geometry_df=None):
    """Create a folium map with property locations"""
    if not MAP_AVAILABLE:
        st.error("Map functionality not available. Install folium and streamlit-folium.")
        return None
    
    # Default center (Spain)
    center_lat, center_lon = 40.4168, -3.7038
    
    # Create the map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add property markers
    if geometry_df is not None and not properties_df.empty:
        added_count = 0
        failed_count = 0
        
        for _, property_row in properties_df.iterrows():
            ref_catastral = property_row.get('referencia_catastral')
            if ref_catastral:
                # Find geometry for this property
                geometry_row = geometry_df[geometry_df['referencia_catastral'] == ref_catastral]
                if not geometry_row.empty:
                    geometry = geometry_row.iloc[0].get('geometry')
                    lat, lon = extract_geometry_center(geometry)
                    
                    if lat and lon and not (np.isnan(lat) or np.isnan(lon)):
                        # Validate coordinates are reasonable for Spain
                        if 35.0 <= lat <= 44.0 and -10.0 <= lon <= 5.0:
                            # Create popup content
                            popup_text = f"""
                            <b>{property_row.get('municipio', 'Unknown')}</b><br>
                            Ref: {ref_catastral}<br>
                            Parcel: {property_row.get('superficie_parcela', 0):,.0f} m²<br>
                            Built: {property_row.get('total_built_area', 0):,.0f} m²<br>
                            Buildings: {int(property_row.get('num_buildings', 0))}<br>
                            Units: {int(property_row.get('num_units', 0))}<br>
                            Coords: {lat:.4f}, {lon:.4f}
                            """
                            
                            # Add marker
                            folium.Marker(
                                location=[lat, lon],
                                popup=folium.Popup(popup_text, max_width=300),
                                tooltip=f"{property_row.get('municipio', 'Unknown')} - {ref_catastral}",
                                icon=folium.Icon(color='blue', icon='home')
                            ).add_to(m)
                            added_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
        
        # Add summary to map
        if added_count > 0:
            st.success(f"📍 Map created successfully: {added_count} properties plotted")
            if failed_count > 0:
                st.info(f"ℹ️ {failed_count} properties could not be plotted (coordinates outside Spain or invalid)")
        else:
            st.warning(f"📍 No properties could be plotted on map ({failed_count} failed)")
            st.info("This might be due to coordinate conversion issues. Properties should still be listed below.")
        
        # Center map on properties if any were added
        if added_count > 0 and added_count <= 100:  # Don't try to fit too many points
            locations = []
            for _, property_row in properties_df.iterrows():
                ref_catastral = property_row.get('referencia_catastral')
                if ref_catastral:
                    geometry_row = geometry_df[geometry_df['referencia_catastral'] == ref_catastral]
                    if not geometry_row.empty:
                        geometry = geometry_row.iloc[0].get('geometry')
                        lat, lon = extract_geometry_center(geometry)
                        if lat and lon and not (np.isnan(lat) or np.isnan(lon)) and 35.0 <= lat <= 44.0 and -10.0 <= lon <= 5.0:
                            locations.append([lat, lon])
            
            if locations:
                # Fit map to show all markers
                m.fit_bounds(locations)
    
    return m

def parse_structured_data(value, data_type='csv'):
    """Parse structured data (CSV or JSON)"""
    if pd.isna(value) or value == '':
        return []
    
    try:
        if data_type == 'csv':
            return [item.strip() for item in str(value).split(',') if item.strip()]
        elif data_type == 'json':
            return json.loads(value)
    except:
        return []

def calculate_match_score(row, filters):
    """Calculate how well a property matches the search criteria"""
    score = 0
    max_score = 0
    
    # Region match (high weight)
    if filters['region'] and filters['region'] != 'All':
        max_score += 30
        if row.get('municipio') == filters['region']:
            score += 30
    
    # Parcel area match (medium weight)
    if filters['parcel_area_range'][0] > 0 or filters['parcel_area_range'][1] < 100000:
        max_score += 20
        parcel_area = row.get('superficie_parcela', 0)
        
        # Handle >100,000 case
        if filters['parcel_area_range'][1] == 100000:
            # If slider at max, check if area >= min value
            if parcel_area >= filters['parcel_area_range'][0]:
                score += 20
            else:
                # Partial score for being close
                distance = abs(parcel_area - filters['parcel_area_range'][0])
                if distance < parcel_area * 0.5:
                    score += 10
        else:
            # Normal range check
            if filters['parcel_area_range'][0] <= parcel_area <= filters['parcel_area_range'][1]:
                score += 20
            else:
                # Partial score for being close
                distance = min(
                    abs(parcel_area - filters['parcel_area_range'][0]),
                    abs(parcel_area - filters['parcel_area_range'][1])
                )
                if distance < parcel_area * 0.5:  # Within 50% of range
                    score += 10
    
    # Built area match (medium weight)
    if filters['built_area_range'][0] > 0 or filters['built_area_range'][1] < 1000:
        max_score += 20
        
        if filters.get('area_search_type', 'Total built area on parcel') == "Total built area on parcel":
            # Traditional total built area scoring
            built_area = row.get('total_built_area', 0)
            
            # Handle >1,000 case
            if filters['built_area_range'][1] == 1000:
                # If slider at max, check if area >= min value
                if built_area >= filters['built_area_range'][0]:
                    score += 20
                else:
                    # Partial score for being close
                    distance = abs(built_area - filters['built_area_range'][0])
                    if distance < built_area * 0.5:
                        score += 10
            else:
                # Normal range check
                if filters['built_area_range'][0] <= built_area <= filters['built_area_range'][1]:
                    score += 20
                else:
                    # Partial score for being close
                    distance = min(
                        abs(built_area - filters['built_area_range'][0]),
                        abs(built_area - filters['built_area_range'][1])
                    )
                    if distance < built_area * 0.5:
                        score += 10
        else:
            # Individual building/unit area scoring
            found_match = False
            
            # Check building areas
            building_areas = parse_structured_data(row.get('buildings_areas', ''), 'csv')
            if building_areas:
                for area_str in building_areas:
                    try:
                        area = float(area_str)
                        if filters['built_area_range'][1] == 1000:
                            if area >= filters['built_area_range'][0]:
                                found_match = True
                                break
                        else:
                            if filters['built_area_range'][0] <= area <= filters['built_area_range'][1]:
                                found_match = True
                                break
                    except (ValueError, TypeError):
                        continue
            
            # Check unit areas if no building match found
            if not found_match:
                unit_areas = parse_structured_data(row.get('units_floor_areas', ''), 'csv')
                if unit_areas:
                    for area_str in unit_areas:
                        try:
                            area = float(area_str)
                            if filters['built_area_range'][1] == 1000:
                                if area >= filters['built_area_range'][0]:
                                    found_match = True
                                    break
                            else:
                                if filters['built_area_range'][0] <= area <= filters['built_area_range'][1]:
                                    found_match = True
                                    break
                        except (ValueError, TypeError):
                            continue
            
            if found_match:
                score += 20
            else:
                # Partial score - check if any area is close to range
                score += 5  # Small bonus for having buildings/units even if not in range
    
    # Year range match (medium weight)
    if filters['year_range'][0] > 1900 or filters['year_range'][1] < 2024:
        max_score += 15
        years_built = parse_structured_data(row.get('units_years_built', ''), 'csv')
        if years_built:
            years = [int(y) for y in years_built if y.isdigit()]
            if years:
                # Check if any building year is in range
                in_range = any(filters['year_range'][0] <= year <= filters['year_range'][1] for year in years)
                if in_range:
                    score += 15
                else:
                    # Partial score for being close
                    min_distance = min([min(abs(year - filters['year_range'][0]), 
                                           abs(year - filters['year_range'][1])) for year in years])
                    if min_distance <= 10:  # Within 10 years
                        score += 7
    
    # Usage type match (medium weight)
    if filters['usage_types']:
        max_score += 15
        primary_use = row.get('primary_use_type', '')
        if primary_use in filters['usage_types']:
            score += 15
        else:
            # Check all use types in structured data
            use_types = parse_structured_data(row.get('units_use_types', ''), 'csv')
            if any(use_type in filters['usage_types'] for use_type in use_types):
                score += 10
    
    # Building count preference (low weight)
    if filters['building_count_range'][0] > 0 or filters['building_count_range'][1] < 50:
        max_score += 10
        building_count = row.get('num_buildings', 0)
        if filters['building_count_range'][0] <= building_count <= filters['building_count_range'][1]:
            score += 10
    

    
    # Return percentage score
    return (score / max_score * 100) if max_score > 0 else 0

def display_property_card(row, rank, match_score):
    """Display a property in a card format"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.metric("Rank", f"#{rank}")
        st.metric("Match Score", f"{match_score:.1f}%")
    
    with col2:
        st.subheader(f"📍 {row.get('municipio', 'Unknown')} - {row.get('referencia_catastral', 'N/A')}")
        
        # Basic info
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Parcel Area", f"{row.get('superficie_parcela', 0):,.0f} m²")
        with col_b:
            st.metric("Built Area", f"{row.get('total_built_area', 0):,.0f} m²")
        with col_c:
            st.metric("Floor Area", f"{row.get('total_floor_area', 0):,.0f} m²")
        
        # Building and unit info
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            st.metric("Buildings", int(row.get('num_buildings', 0)))
        with col_e:
            st.metric("Units", int(row.get('num_units', 0)))
        with col_f:
            utilization = row.get('utilization_score', 0)
            st.metric("Utilization", f"{utilization:.2f}")
    
    with col3:
        if st.button(f"🔍 View Details", key=f"details_{rank}_{row.get('referencia_catastral', rank)}"):
            st.session_state[f'show_details_{rank}'] = True
    
    # Show detailed information if requested
    if st.session_state.get(f'show_details_{rank}', False):
        with st.expander(f"📋 Detailed Information - {row.get('referencia_catastral', 'N/A')}", expanded=True):
            
            # Building details
            st.subheader("🏢 Buildings")
            buildings_areas = parse_structured_data(row.get('buildings_areas', ''), 'csv')
            buildings_types = parse_structured_data(row.get('buildings_types', ''), 'csv')
            
            if buildings_areas and buildings_types:
                building_df = pd.DataFrame({
                    'Area (m²)': buildings_areas[:len(buildings_types)],
                    'Type': buildings_types
                })
                st.dataframe(building_df, use_container_width=True)
            else:
                st.info("No building details available")
            
            # Unit details
            st.subheader("🏠 Units")
            units_years = parse_structured_data(row.get('units_years_built', ''), 'csv')
            units_ages = parse_structured_data(row.get('units_ages', ''), 'csv')
            units_types = parse_structured_data(row.get('units_use_types', ''), 'csv')
            
            if units_years and units_types:
                unit_df = pd.DataFrame({
                    'Year Built': units_years[:len(units_types)],
                    'Age': units_ages[:len(units_types)] if units_ages else ['N/A'] * len(units_types),
                    'Use Type': units_types
                })
                st.dataframe(unit_df, use_container_width=True)
            else:
                st.info("No unit details available")
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Primary Building Type", row.get('primary_building_type', 'N/A'))
            with col2:
                st.metric("Primary Use Type", row.get('primary_use_type', 'N/A'))
            with col3:
                residential_ratio = row.get('residential_ratio', 0)
                st.metric("Residential %", f"{residential_ratio*100:.1f}%")
            with col4:
                avg_age = row.get('avg_building_age', 0)
                st.metric("Avg Building Age", f"{avg_age:.1f} years")
            
            if st.button(f"❌ Close Details", key=f"close_{rank}_{row.get('referencia_catastral', rank)}"):
                st.session_state[f'show_details_{rank}'] = False
                st.rerun()
    
    st.divider()

def main():
    st.title("🏠 Catastro Property Search System")
    st.markdown("Search and explore Spanish cadastral properties with advanced filtering and ranking")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Load geometry data for mapping
    geometry_df = None
    if MAP_AVAILABLE:
        geometry_df = load_full_data_with_geometry()
        if geometry_df is not None:
            st.sidebar.success(f"📍 Map ready! Loaded {len(geometry_df)} parcels with geometry")
        else:
            st.sidebar.error("📍 Failed to load geometry data")
    else:
        st.sidebar.warning("📍 Map requires: `pip install folium streamlit-folium`")
    
    # Sidebar filters
    st.sidebar.header("🔍 Search Filters")
    
    # Region filter
    regions = ['All'] + sorted(df['municipio'].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox("📍 Region (Municipality)", regions)
    
    # Parcel area filter (0 to 100,000, with >100,000 option)
    parcel_area_range = st.sidebar.slider(
        "📏 Parcel Area (m²)", 
        0, 100000, 
        (0, 100000),
        step=1000
    )
    
    # Exact number inputs for parcel area
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Exact Values:**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        parcel_min_exact = st.number_input(
            "Min (m²)", 
            min_value=0, 
            max_value=1000000, 
            value=parcel_area_range[0],
            step=100,
            key="parcel_min"
        )
    with col2:
        parcel_max_exact = st.number_input(
            "Max (m²)", 
            min_value=0, 
            max_value=1000000, 
            value=parcel_area_range[1] if parcel_area_range[1] < 100000 else 100000,
            step=100,
            key="parcel_max"
        )
        # Checkbox for >100,000
        parcel_unlimited = st.checkbox("No max limit", value=(parcel_area_range[1] == 100000), key="parcel_unlimited")
    
    # Use exact values if provided, otherwise use slider values
    if parcel_unlimited:
        final_parcel_range = (parcel_min_exact, 100000)  # Use 100000 as marker for unlimited
        parcel_display = f"{parcel_min_exact:,} - >100,000 m²"
    else:
        final_parcel_range = (parcel_min_exact, parcel_max_exact)
        parcel_display = f"{parcel_min_exact:,} - {parcel_max_exact:,} m²"
    
    st.sidebar.caption(f"Selected: {parcel_display}")
    
    # Built area filter (0 to 1,000, with >1,000 option)
    # Option to search total or individual building/unit area
    area_search_type = st.sidebar.radio(
        "🏗️ Built Area Search Type:",
        ["Total built area on parcel", "Individual building/unit area"],
        help="Choose whether to search by total area or individual building/unit size"
    )
    
    built_area_range = st.sidebar.slider(
        "Built Area (m²)", 
        0, 1000, 
        (0, 1000),
        step=50
    )
    
    # Exact number inputs for built area
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Exact Values:**")
    col3, col4 = st.sidebar.columns(2)
    with col3:
        built_min_exact = st.number_input(
            "Min (m²)", 
            min_value=0, 
            max_value=100000, 
            value=built_area_range[0],
            step=10,
            key="built_min"
        )
    with col4:
        built_max_exact = st.number_input(
            "Max (m²)", 
            min_value=0, 
            max_value=100000, 
            value=built_area_range[1] if built_area_range[1] < 1000 else 1000,
            step=10,
            key="built_max"
        )
        # Checkbox for >1,000
        built_unlimited = st.checkbox("No max limit", value=(built_area_range[1] == 1000), key="built_unlimited")
    
    # Use exact values if provided, otherwise use slider values
    if built_unlimited:
        final_built_range = (built_min_exact, 1000)  # Use 1000 as marker for unlimited
        built_display = f"{built_min_exact:,} - >1,000 m²"
    else:
        final_built_range = (built_min_exact, built_max_exact)
        built_display = f"{built_min_exact:,} - {built_max_exact:,} m²"
    
    st.sidebar.caption(f"Selected: {built_display}")
    
    st.sidebar.markdown("---")
    
    # Year range filter
    current_year = datetime.now().year
    year_range = st.sidebar.slider(
        "📅 Year Built Range", 
        1900, current_year, 
        (1970, current_year),
        step=5
    )
    
    # Usage type filter
    available_usage_types = []
    for usage_str in df['units_use_types'].dropna():
        usage_types = parse_structured_data(usage_str, 'csv')
        available_usage_types.extend(usage_types)
    unique_usage_types = sorted(list(set(available_usage_types)))
    
    selected_usage_types = st.sidebar.multiselect(
        "🏘️ Usage Types", 
        unique_usage_types,
        default=[]
    )
    
    # Building count filter
    max_buildings = int(df['num_buildings'].max()) if df['num_buildings'].max() > 0 else 10
    building_count_range = st.sidebar.slider(
        "🏢 Number of Buildings", 
        0, max_buildings, 
        (0, max_buildings),
        step=1
    )
    

    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        max_results = st.number_input("📊 Maximum Results", 1, 100, 20)
        min_match_score = st.slider("🎯 Minimum Match Score (%)", 0, 100, 0)
    
    # Compile filters
    filters = {
        'region': selected_region,
        'parcel_area_range': final_parcel_range,
        'built_area_range': final_built_range,
        'area_search_type': area_search_type,
        'year_range': year_range,
        'usage_types': selected_usage_types,
        'building_count_range': building_count_range
    }
    
    # Search button
    if st.sidebar.button("🔍 Search Properties", type="primary"):
        
        # Apply basic filters
        filtered_df = df.copy()
        
        if selected_region != 'All':
            filtered_df = filtered_df[filtered_df['municipio'] == selected_region]
        
        # Parcel area filtering with exact values and unlimited option
        if final_parcel_range[0] > 0 or final_parcel_range[1] < 100000:
            if parcel_unlimited or final_parcel_range[1] == 100000:
                # When unlimited is checked, include all parcels >= min value
                filtered_df = filtered_df[filtered_df['superficie_parcela'] >= final_parcel_range[0]]
            else:
                # Normal range filtering with exact values
                filtered_df = filtered_df[
                    (filtered_df['superficie_parcela'] >= final_parcel_range[0]) & 
                    (filtered_df['superficie_parcela'] <= final_parcel_range[1])
                ]
        
        # Built area filtering with exact values and unlimited option
        if final_built_range[0] > 0 or final_built_range[1] < 1000:
            if area_search_type == "Total built area on parcel":
                # Traditional total built area filtering
                if built_unlimited or final_built_range[1] == 1000:
                    # When unlimited is checked, include all built areas >= min value
                    filtered_df = filtered_df[filtered_df['total_built_area'] >= final_built_range[0]]
                else:
                    # Normal range filtering with exact values
                    filtered_df = filtered_df[
                        (filtered_df['total_built_area'] >= final_built_range[0]) & 
                        (filtered_df['total_built_area'] <= final_built_range[1])
                    ]
            else:
                # Individual building/unit area filtering
                def has_individual_area_in_range(row):
                    # Check building areas
                    building_areas = parse_structured_data(row.get('buildings_areas', ''), 'csv')
                    if building_areas:
                        for area_str in building_areas:
                            try:
                                area = float(area_str)
                                if built_unlimited or final_built_range[1] == 1000:
                                    if area >= final_built_range[0]:
                                        return True
                                else:
                                    if final_built_range[0] <= area <= final_built_range[1]:
                                        return True
                            except (ValueError, TypeError):
                                continue
                    
                    # Check unit areas (floor areas)
                    unit_areas = parse_structured_data(row.get('units_floor_areas', ''), 'csv')
                    if unit_areas:
                        for area_str in unit_areas:
                            try:
                                area = float(area_str)
                                if built_unlimited or final_built_range[1] == 1000:
                                    if area >= final_built_range[0]:
                                        return True
                                else:
                                    if final_built_range[0] <= area <= final_built_range[1]:
                                        return True
                            except (ValueError, TypeError):
                                continue
                    
                    return False
                
                filtered_df = filtered_df[filtered_df.apply(has_individual_area_in_range, axis=1)]
        
        # Building count filtering
        max_buildings = int(df['num_buildings'].max()) if df['num_buildings'].max() > 0 else 10
        if building_count_range[0] > 0 or building_count_range[1] < max_buildings:
            filtered_df = filtered_df[
                (filtered_df['num_buildings'] >= building_count_range[0]) & 
                (filtered_df['num_buildings'] <= building_count_range[1])
            ]
        
        # Calculate match scores
        if not filtered_df.empty:
            filtered_df['match_score'] = filtered_df.apply(
                lambda row: calculate_match_score(row, filters), axis=1
            )
            
            # Filter by minimum match score
            filtered_df = filtered_df[filtered_df['match_score'] >= min_match_score]
            
            # Sort by match score
            filtered_df = filtered_df.sort_values('match_score', ascending=False)
            
            # Limit results
            filtered_df = filtered_df.head(max_results)
            
            st.session_state['search_results'] = filtered_df
            st.session_state['search_performed'] = True
        else:
            st.session_state['search_results'] = pd.DataFrame()
            st.session_state['search_performed'] = True
    
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
            if MAP_AVAILABLE and geometry_df is not None:
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
                    map_obj = create_map(map_df, geometry_df)
                    if map_obj:
                        map_data = st_folium(map_obj, width=700, height=400)
                        
                        # Show selected property info if clicked
                        if map_data.get('last_object_clicked_popup'):
                            st.info("💡 Click on map markers to see property details!")
                        
                        # Debug info
                        if st.checkbox("Show debug info", value=False):
                            st.write(f"**Debug Information:**")
                            st.write(f"- Geometry DF shape: {geometry_df.shape if geometry_df is not None else 'None'}")
                            st.write(f"- Search results count: {len(map_df)}")
                            st.write(f"- MAP_AVAILABLE: {MAP_AVAILABLE}")
                            if geometry_df is not None and not geometry_df.empty:
                                st.write(f"- Available geometry columns: {list(geometry_df.columns)}")
                                
                                # Test coordinate extraction on first result
                                if not map_df.empty:
                                    test_ref = map_df.iloc[0]['referencia_catastral']
                                    test_geom_row = geometry_df[geometry_df['referencia_catastral'] == test_ref]
                                    if not test_geom_row.empty:
                                        test_geom = test_geom_row.iloc[0].get('geometry')
                                        test_lat, test_lon = extract_geometry_center(test_geom)
                                        st.write(f"- Test coordinate extraction for {test_ref}: {test_lat}, {test_lon}")
                                    else:
                                        st.write(f"- ❌ No geometry found for test property {test_ref}")
                    else:
                        st.error("Failed to create map")
            else:
                if MAP_AVAILABLE:
                    st.info("📍 Map not available - geometry data not found")
                else:
                    st.info("📍 Map requires: `pip install folium streamlit-folium pyproj`")
            
            st.divider()
            
            # Display each property
            for idx, (_, row) in enumerate(results_df.iterrows(), 1):
                display_property_card(row, idx, row['match_score'])
            
            # Download results
            st.subheader("📥 Download Results")
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Search Results as CSV",
                data=csv,
                file_name=f"catastro_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("🔍 No properties found matching your criteria. Try adjusting the filters.")
    
    else:
        # Show sample data and instructions
        st.header("🏠 Welcome to Catastro Property Search")
        st.markdown("""
        Use the filters in the sidebar to search for properties that match your criteria:
        
        - **📍 Region**: Select specific municipality
        - **📏 Parcel Area**: Filter by land size (0-100,000+ m²)
        - **🏗️ Built Area**: Filter by construction size (0-1,000+ m²)
        - **📅 Year Built**: Find properties from specific time periods
        - **🏘️ Usage Types**: Residential, Industrial, etc.
        - **🏢 Building Count**: Number of buildings on parcel
        
        Properties are ranked by **match score** - how well they fit your criteria!
        """)
        
        # Show data overview
        st.subheader("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Properties", f"{len(df):,}")
        with col2:
            st.metric("Municipalities", df['municipio'].nunique())
        with col3:
            st.metric("Total Land Area", f"{df['superficie_parcela'].sum():,.0f} m²")
        with col4:
            st.metric("Total Built Area", f"{df['total_built_area'].sum():,.0f} m²")
        
        # Sample properties
        st.subheader("📋 Sample Properties")
        sample_df = df.sample(min(5, len(df)))[['referencia_catastral', 'municipio', 'superficie_parcela', 'total_built_area', 'num_buildings', 'num_units']]
        st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main() 