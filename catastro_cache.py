#!/usr/bin/env python3
"""
Caching layer for Catastro database queries
Provides efficient caching with TTL and invalidation
"""

import streamlit as st
import pandas as pd
import hashlib
import json
import time
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CatastroCache:
    """Cache manager for Catastro database queries"""
    
    def __init__(self, default_ttl: int = 3600):  # 1 hour default TTL
        self.default_ttl = default_ttl
        self.cache_key_prefix = "catastro_cache_"
        
    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key for operation with parameters"""
        # Create a consistent hash of the parameters
        params_str = json.dumps(kwargs, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"{self.cache_key_prefix}{operation}_{params_hash}"
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        
        expiry_time = cache_entry.get('expiry_time')
        if not expiry_time:
            return False
        
        return datetime.now() < expiry_time
    
    def get_cached_data(self, operation: str, **kwargs) -> Optional[pd.DataFrame]:
        """Get cached data if available and valid"""
        cache_key = self._get_cache_key(operation, **kwargs)
        
        if cache_key in st.session_state:
            cache_entry = st.session_state[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.info(f"Cache hit for {operation}")
                return cache_entry['data']
            else:
                # Remove expired cache
                del st.session_state[cache_key]
                logger.info(f"Cache expired for {operation}")
        
        logger.info(f"Cache miss for {operation}")
        return None
    
    def set_cached_data(self, operation: str, data: pd.DataFrame, ttl: Optional[int] = None, **kwargs) -> None:
        """Cache data with TTL"""
        cache_key = self._get_cache_key(operation, **kwargs)
        ttl = ttl or self.default_ttl
        
        cache_entry = {
            'data': data,
            'cached_at': datetime.now(),
            'expiry_time': datetime.now() + timedelta(seconds=ttl),
            'operation': operation,
            'kwargs': kwargs
        }
        
        st.session_state[cache_key] = cache_entry
        logger.info(f"Cached data for {operation} (TTL: {ttl}s)")
    
    def invalidate_cache(self, operation: Optional[str] = None) -> None:
        """Invalidate cache for specific operation or all cache"""
        if operation:
            # Invalidate specific operation
            keys_to_remove = [key for key in st.session_state.keys() 
                            if key.startswith(self.cache_key_prefix) and operation in key]
        else:
            # Invalidate all cache
            keys_to_remove = [key for key in st.session_state.keys() 
                            if key.startswith(self.cache_key_prefix)]
        
        for key in keys_to_remove:
            del st.session_state[key]
        
        logger.info(f"Invalidated cache for {operation if operation else 'all operations'}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state"""
        cache_keys = [key for key in st.session_state.keys() 
                     if key.startswith(self.cache_key_prefix)]
        
        info = {
            'total_cached_items': len(cache_keys),
            'cache_details': []
        }
        
        for key in cache_keys:
            cache_entry = st.session_state[key]
            info['cache_details'].append({
                'operation': cache_entry.get('operation', 'Unknown'),
                'cached_at': cache_entry.get('cached_at'),
                'expiry_time': cache_entry.get('expiry_time'),
                'is_valid': self._is_cache_valid(cache_entry),
                'data_shape': cache_entry['data'].shape if 'data' in cache_entry else None
            })
        
        return info

# Global cache instance
catastro_cache = CatastroCache()

# Streamlit cache decorators for database operations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_search_ready_data():
    """Cached version of search-ready data"""
    from database_service import catastro_db
    return catastro_db.get_search_ready_data()

@st.cache_data(ttl=7200)  # Cache for 2 hours (geometry data changes less frequently)
def cached_geometry_data():
    """Cached version of geometry data"""
    from database_service import catastro_db
    return catastro_db.get_geometry_data()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def cached_comprehensive_data():
    """Cached version of comprehensive data"""
    from database_service import catastro_db
    return catastro_db.process_comprehensive_data()

def clear_all_caches():
    """Clear all Streamlit caches"""
    cached_search_ready_data.clear()
    cached_geometry_data.clear()
    cached_comprehensive_data.clear()
    catastro_cache.invalidate_cache()
    logger.info("All caches cleared")

def get_cache_status():
    """Get current cache status for debugging"""
    return {
        'streamlit_cache_info': {
            'search_ready_data_cached': len(cached_search_ready_data._mem_cache) > 0,
            'geometry_data_cached': len(cached_geometry_data._mem_cache) > 0,
            'comprehensive_data_cached': len(cached_comprehensive_data._mem_cache) > 0
        },
        'custom_cache_info': catastro_cache.get_cache_info()
    } 