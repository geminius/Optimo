"""
Cache Service for API performance optimization.

This module provides a caching layer for frequently accessed data
to reduce database queries and improve API response times.
"""

import logging
import threading
import time
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached value with metadata."""
    value: Any
    created_at: float
    ttl_seconds: float
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds


class CacheService:
    """
    Thread-safe caching service for API data.
    
    Provides in-memory caching with TTL (time-to-live) support
    and automatic cache invalidation. Designed for caching:
    - Dashboard statistics
    - Configuration data
    - Session lists
    - Model metadata
    
    Features:
    - Thread-safe operations
    - TTL-based expiration
    - Manual cache invalidation
    - Cache statistics tracking
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern to ensure single instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the cache service."""
        # Only initialize once
        if self._initialized:
            return
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        
        # Thread safety for cache operations
        self._cache_lock = threading.RLock()
        
        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._invalidations = 0
        
        self._initialized = True
        self.logger.info("CacheService initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if exists and not expired, None otherwise
        """
        with self._cache_lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                self.logger.debug(f"Cache miss: {key}")
                return None
            
            if entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                self._misses += 1
                self.logger.debug(f"Cache expired: {key}")
                return None
            
            self._hits += 1
            self.logger.debug(f"Cache hit: {key}")
            return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: float = 60.0) -> None:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (default: 60)
        """
        with self._cache_lock:
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl_seconds
            )
            self._cache[key] = entry
            self.logger.debug(f"Cache set: {key} (TTL: {ttl_seconds}s)")
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate (remove) cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was removed, False if not found
        """
        with self._cache_lock:
            if key in self._cache:
                del self._cache[key]
                self._invalidations += 1
                self.logger.debug(f"Cache invalidated: {key}")
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching pattern.
        
        Args:
            pattern: Pattern to match (simple substring match)
            
        Returns:
            Number of entries invalidated
        """
        with self._cache_lock:
            keys_to_remove = [
                key for key in self._cache.keys()
                if pattern in key
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
                self._invalidations += 1
            
            if keys_to_remove:
                self.logger.debug(
                    f"Cache invalidated {len(keys_to_remove)} entries matching: {pattern}"
                )
            
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
            self._invalidations += count
            self.logger.info(f"Cache cleared: {count} entries removed")
    
    def get_or_set(
        self, 
        key: str, 
        factory: Callable[[], Any], 
        ttl_seconds: float = 60.0
    ) -> Any:
        """
        Get value from cache or compute and cache it.
        
        This is a convenience method that combines get and set operations.
        If the value is in cache and not expired, it's returned immediately.
        Otherwise, the factory function is called to compute the value,
        which is then cached and returned.
        
        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl_seconds: Time-to-live in seconds (default: 60)
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache first
        value = self.get(key)
        if value is not None:
            return value
        
        # Compute value
        value = factory()
        
        # Cache and return
        self.set(key, value, ttl_seconds)
        return value
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired cache entries.
        
        Returns:
            Number of entries removed
        """
        with self._cache_lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._cache_lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
            
            return {
                "total_entries": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": hit_rate,
                "invalidations": self._invalidations,
                "total_requests": total_requests
            }
    
    def reset_statistics(self) -> None:
        """Reset cache statistics counters."""
        with self._cache_lock:
            self._hits = 0
            self._misses = 0
            self._invalidations = 0
            self.logger.debug("Cache statistics reset")
    
    @staticmethod
    def invalidate_dashboard_cache() -> None:
        """
        Convenience method to invalidate dashboard-related caches.
        
        This should be called when:
        - A new optimization session is created
        - A session status changes
        - A session completes or fails
        - Models are added or removed
        """
        cache = CacheService()
        cache.invalidate("dashboard:stats")
        logger.debug("Dashboard cache invalidated")
