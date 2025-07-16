#!/usr/bin/env python3
"""
Performance Profiler for Catastro Database Operations
Identifies bottlenecks in data loading and aggregation processes
"""

import time
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Profile performance of database operations"""
    
    def __init__(self):
        self.timings = {}
        self.data_sizes = {}
        
    def time_operation(self, operation_name: str, func, *args, **kwargs):
        """Time a specific operation"""
        print(f"\n⏱️  Starting: {operation_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            self.timings[operation_name] = duration
            
            # Record data size if result is a DataFrame
            if isinstance(result, pd.DataFrame):
                self.data_sizes[operation_name] = {
                    'rows': len(result),
                    'columns': len(result.columns),
                    'memory_mb': round(result.memory_usage(deep=True).sum() / 1024**2, 2)
                }
            
            print(f"✅ Completed: {operation_name} in {duration:.2f} seconds")
            if isinstance(result, pd.DataFrame):
                print(f"   Data size: {len(result):,} rows, {len(result.columns)} columns")
                print(f"   Memory usage: {self.data_sizes[operation_name]['memory_mb']} MB")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"❌ Failed: {operation_name} after {duration:.2f} seconds - {e}")
            raise
    
    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        
        # Sort by duration
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        total_time = sum(self.timings.values())
        
        print(f"\n🕐 Total Time: {total_time:.2f} seconds")
        print(f"📈 Operations: {len(self.timings)}")
        
        print(f"\n⏱️  Individual Timings:")
        for operation, duration in sorted_timings:
            percentage = (duration / total_time) * 100
            print(f"   {operation:<35} {duration:>8.2f}s ({percentage:>5.1f}%)")
            
            if operation in self.data_sizes:
                data_info = self.data_sizes[operation]
                print(f"   {'':>35} {data_info['rows']:,} rows, {data_info['memory_mb']} MB")
        
        print(f"\n🔍 Analysis:")
        if sorted_timings:
            slowest_op, slowest_time = sorted_timings[0]
            print(f"   Slowest operation: {slowest_op} ({slowest_time:.2f}s)")
            
            # Identify bottleneck type
            db_operations = [op for op, _ in sorted_timings if 'loading' in op.lower() or 'connection' in op.lower()]
            processing_operations = [op for op, _ in sorted_timings if 'aggregating' in op.lower() or 'processing' in op.lower()]
            
            db_time = sum(self.timings[op] for op in db_operations)
            processing_time = sum(self.timings[op] for op in processing_operations)
            
            print(f"   Database operations: {db_time:.2f}s ({db_time/total_time*100:.1f}%)")
            print(f"   Processing operations: {processing_time:.2f}s ({processing_time/total_time*100:.1f}%)")
            
            if db_time > processing_time:
                print(f"   💡 Bottleneck: Database I/O (consider query optimization)")
            else:
                print(f"   💡 Bottleneck: Data processing (consider algorithm optimization)")

def profile_database_operations():
    """Profile all database operations step by step"""
    profiler = PerformanceProfiler()
    
    print("🔍 PROFILING DATABASE OPERATIONS")
    print("="*60)
    
    try:
        from database_service import catastro_db
        
        # 1. Test basic connection
        def test_connection():
            return catastro_db.test_connection()
        
        profiler.time_operation("Database Connection Test", test_connection)
        
        # 2. Load each dataset separately
        df_parcels = profiler.time_operation("Loading Parcels Data", catastro_db.load_parcels_data)
        df_buildings = profiler.time_operation("Loading Buildings Data", catastro_db.load_buildings_data)
        df_units = profiler.time_operation("Loading Units Data", catastro_db.load_units_data)
        
        # 3. Test aggregation operations separately
        building_agg = profiler.time_operation("Aggregating Buildings", catastro_db.aggregate_buildings_data, df_buildings)
        unit_agg = profiler.time_operation("Aggregating Units", catastro_db.aggregate_units_data, df_units)
        
        # 4. Test merging operations
        def merge_data():
            final_df = df_parcels.copy()
            final_df = final_df.merge(building_agg, left_on='referencia_catastral', right_on='parcel_ref', how='left')
            final_df = final_df.merge(unit_agg, left_on='referencia_catastral', right_on='parcel_ref', how='left')
            return final_df
        
        merged_df = profiler.time_operation("Merging Data", merge_data)
        
        # 5. Test computed metrics
        final_df = profiler.time_operation("Computing Metrics", catastro_db.add_computed_metrics, merged_df)
        
        # 6. Test geometry loading separately
        geometry_df = profiler.time_operation("Loading Geometry Data", catastro_db.get_geometry_data)
        
        profiler.print_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def profile_specific_queries():
    """Profile specific slow queries"""
    profiler = PerformanceProfiler()
    
    print(f"\n🔍 PROFILING SPECIFIC QUERIES")
    print("="*60)
    
    try:
        from database_service import catastro_db
        
        # Test individual table queries with row limits
        def test_limited_parcels():
            query = f"SELECT TOP 1000 * FROM {catastro_db.PARCEL_TABLE}"
            return catastro_db.execute_query(query)
        
        def test_limited_buildings():
            query = f"SELECT TOP 1000 * FROM {catastro_db.BUILDING_TABLE}"
            return catastro_db.execute_query(query)
            
        def test_limited_units():
            query = f"SELECT TOP 1000 * FROM {catastro_db.UNIT_TABLE}"
            return catastro_db.execute_query(query)
        
        # Test small datasets
        profiler.time_operation("Loading 1000 Parcels", test_limited_parcels)
        profiler.time_operation("Loading 1000 Buildings", test_limited_buildings)
        profiler.time_operation("Loading 1000 Units", test_limited_units)
        
        # Test count queries (should be fast)
        def count_parcels():
            query = f"SELECT COUNT(*) FROM {catastro_db.PARCEL_TABLE}"
            return catastro_db.execute_query(query)
            
        def count_buildings():
            query = f"SELECT COUNT(*) FROM {catastro_db.BUILDING_TABLE}"
            return catastro_db.execute_query(query)
            
        def count_units():
            query = f"SELECT COUNT(*) FROM {catastro_db.UNIT_TABLE}"
            return catastro_db.execute_query(query)
        
        profiler.time_operation("Count Parcels", count_parcels)
        profiler.time_operation("Count Buildings", count_buildings)
        profiler.time_operation("Count Units", count_units)
        
        profiler.print_summary()
        
    except Exception as e:
        print(f"❌ Query profiling failed: {e}")
        import traceback
        traceback.print_exc()

def compare_optimization_strategies():
    """Compare different optimization strategies"""
    print(f"\n🚀 TESTING OPTIMIZATION STRATEGIES")
    print("="*60)
    
    try:
        from database_service import catastro_db
        
        # Strategy 1: Load only essential columns
        def load_essential_parcels():
            query = f"""
            SELECT 
                referencia_catastral,
                municipio,
                provincia,
                superficie_parcela
            FROM {catastro_db.PARCEL_TABLE}
            """
            return catastro_db.execute_query(query)
        
        def load_essential_buildings():
            query = f"""
            SELECT 
                parcel_ref,
                building_type,
                built_area
            FROM {catastro_db.BUILDING_TABLE}
            """
            return catastro_db.execute_query(query)
        
        def load_essential_units():
            query = f"""
            SELECT 
                parcel_ref,
                use_type,
                floor_area,
                year_built
            FROM {catastro_db.UNIT_TABLE}
            """
            return catastro_db.execute_query(query)
        
        profiler = PerformanceProfiler()
        
        # Test optimized loading
        profiler.time_operation("Load Essential Parcels", load_essential_parcels)
        profiler.time_operation("Load Essential Buildings", load_essential_buildings)
        profiler.time_operation("Load Essential Units", load_essential_units)
        
        profiler.print_summary()
        
        print(f"\n💡 Optimization Recommendations:")
        print(f"   1. Load only essential columns for search functionality")
        print(f"   2. Consider pagination for large datasets")
        print(f"   3. Use database-side aggregation where possible")
        print(f"   4. Implement incremental loading")
        
    except Exception as e:
        print(f"❌ Optimization testing failed: {e}")

if __name__ == "__main__":
    print("🔍 CATASTRO PERFORMANCE PROFILER")
    print("="*60)
    
    # Profile basic operations
    success = profile_database_operations()
    
    if success:
        # Profile specific queries
        profile_specific_queries()
        
        # Test optimization strategies
        compare_optimization_strategies()
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   1. Check the performance summary above")
        print(f"   2. Focus optimization on the slowest operations")
        print(f"   3. Consider implementing lazy loading for large datasets")
        print(f"   4. Use database indexes on frequently queried columns")
        
    else:
        print(f"❌ Could not complete profiling due to errors") 