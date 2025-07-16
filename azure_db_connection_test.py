#!/usr/bin/env python3
"""
Azure SQL Database Connection Test Script
Tests connection to the hellodata database and verifies table access.
"""

import pyodbc
import sys
from datetime import datetime

# Database connection parameters
server = "hellodata-database.database.windows.net"
database = "hellodata"
username = "hellodata_prod"
password = "Dvd^1i83]70q"
driver = "ODBC Driver 18 for SQL Server"

# Table names to test
PARCEL_TABLE = "catastro_parcels"
UNIT_TABLE = "catastro_units"
BUILDING_TABLE = "catastro_buildings"

def test_connection(actual_driver):
    """Test the database connection and basic operations."""
    print("=" * 60)
    print("Azure SQL Database Connection Test")
    print("=" * 60)
    print(f"Server: {server}")
    print(f"Database: {database}")
    print(f"Username: {username}")
    print(f"Driver: {actual_driver}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Build connection string
    conn_str = f"DRIVER={actual_driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    
    try:
        print("Attempting to connect to Azure SQL Database...")
        
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            print("✅ Connection successful!")
            
            # Test basic database info
            print("\n📊 Database Information:")
            cursor.execute("SELECT @@VERSION")
            version = cursor.fetchone()[0]
            print(f"SQL Server Version: {version}")
            
            cursor.execute("SELECT DB_NAME()")
            db_name = cursor.fetchone()[0]
            print(f"Connected to database: {db_name}")
            
            # Test table existence and basic info
            print("\n🔍 Testing table access...")
            tables_to_check = [PARCEL_TABLE, UNIT_TABLE, BUILDING_TABLE]
            
            for table_name in tables_to_check:
                try:
                    # Check if table exists and get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    print(f"✅ Table '{table_name}': {row_count:,} rows")
                    
                    # Get column information
                    cursor.execute(f"""
                        SELECT COLUMN_NAME, DATA_TYPE 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_NAME = '{table_name}'
                        ORDER BY ORDINAL_POSITION
                    """)
                    columns = cursor.fetchall()
                    column_info = [f"{col[0]} ({col[1]})" for col in columns[:5]]  # Show first 5 columns
                    print(f"   Columns (first 5): {', '.join(column_info)}")
                    
                except Exception as e:
                    print(f"❌ Error accessing table '{table_name}': {str(e)}")
            
            # Test a simple query on each table
            print("\n🔄 Testing sample queries...")
            for table_name in tables_to_check:
                try:
                    cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
                    row = cursor.fetchone()
                    if row:
                        print(f"✅ Sample query on '{table_name}': Retrieved first row successfully")
                    else:
                        print(f"⚠️  Table '{table_name}' is empty")
                except Exception as e:
                    print(f"❌ Error querying table '{table_name}': {str(e)}")
            
            print("\n🎉 Connection test completed successfully!")
            
    except pyodbc.Error as e:
        print(f"❌ Database connection failed!")
        print(f"Error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False
    
    return True

def check_driver_availability():
    """Check if the required ODBC driver is available."""
    print("\n🔧 Checking ODBC driver availability...")
    drivers = pyodbc.drivers()
    
    if driver in drivers:
        print(f"✅ Required driver '{driver}' is available")
        return driver
    else:
        print(f"❌ Required driver '{driver}' is NOT available")
        
        # Check for fallback drivers
        fallback_drivers = ["SQL Server", "ODBC Driver 17 for SQL Server", "ODBC Driver 13 for SQL Server"]
        
        for fallback in fallback_drivers:
            if fallback in drivers:
                print(f"✅ Found fallback driver: '{fallback}'")
                print("⚠️  Note: Using fallback driver may have limited functionality")
                return fallback
        
        print("❌ No compatible ODBC drivers found")
        print("Available drivers:")
        for d in drivers:
            print(f"  - {d}")
        print("\nPlease install the Microsoft ODBC Driver 18 for SQL Server")
        return None

if __name__ == "__main__":
    print("Starting Azure SQL Database connection test...\n")
    
    # Check driver availability first
    available_driver = check_driver_availability()
    if not available_driver:
        sys.exit(1)
    
    # Test the connection
    if test_connection(available_driver):
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Connection test failed!")
        sys.exit(1) 