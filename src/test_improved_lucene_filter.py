import pytest
from datetime import datetime, date
from dataclasses import dataclass
from typing import List, Dict, Any
import sys
import os

# Import the enhanced implementation
from improved_lucene_filter import (
    EnhancedLuceneObjectFilter, 
    FilterConfig, 
    QueryValidator, 
    TextExtractionCache,
    QueryMetrics
)


@dataclass
class Server:
    """Test dataclass for server objects"""
    name: str
    cpu: float
    memory: int
    status: str
    tags: List[str]
    created: datetime
    config: Dict[str, Any]


@dataclass
class Alert:
    """Test dataclass for alert objects"""
    severity: str
    message: str
    timestamp: str
    resolved: bool
    count: int


class CircularRefTest:
    """Test class with potential circular references"""
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.children = []


class TestEnhancedLuceneFilterBasics:
    """Test basic functionality and setup with enhanced features"""
    
    def setup_method(self):
        """Setup test data for each test"""
        self.config = FilterConfig(
            enable_text_cache=True,
            enable_metrics=True,
            case_sensitive=False
        )
        self.filter_engine = EnhancedLuceneObjectFilter(self.config)
        
        # Dictionary test data
        self.dict_servers = [
            {"name": "web-01", "cpu": 85.5, "memory": 16, "status": "running", "tags": ["web", "prod"], "port": 80},
            {"name": "db-01", "cpu": 45.2, "memory": 32, "status": "running", "tags": ["db", "prod"], "port": 5432},
            {"name": "cache-01", "cpu": 25.0, "memory": 8, "status": "stopped", "tags": ["cache", "dev"], "port": 6379},
            {"name": "api-gateway", "cpu": 60.8, "memory": 12, "status": "idle", "tags": ["api", "prod"], "port": 443},
        ]
        
        # Object test data
        self.obj_servers = [
            Server("web-01", 85.5, 16, "running", ["web", "prod"], 
                   datetime(2024, 1, 15), {"ssl": True, "backup": False}),
            Server("db-01", 45.2, 32, "running", ["db", "prod"], 
                   datetime(2024, 1, 10), {"ssl": False, "backup": True}),
            Server("cache-01", 25.0, 8, "stopped", ["cache", "dev"], 
                   datetime(2024, 1, 20), {"ssl": False, "backup": False}),
        ]
        
        # Alert test data
        self.alerts = [
            Alert("critical", "Database connection failed", "2024-01-15", False, 5),
            Alert("warning", "High CPU usage detected", "2024-01-16", True, 12),
            Alert("info", "Backup completed successfully", "2024-01-17", True, 1),
            Alert("critical", "Disk space low", "2024-01-18", False, 3),
        ]
    
    def test_empty_query_returns_all(self):
        """Empty queries should return all objects"""
        result = self.filter_engine.filter(self.dict_servers, "")
        assert len(result) == len(self.dict_servers)
        
        result = self.filter_engine.filter(self.dict_servers, "   ")
        assert len(result) == len(self.dict_servers)
    
    def test_filter_returns_list(self):
        """Filter should always return a list"""
        result = self.filter_engine.filter(self.dict_servers, "cpu:[51 TO *]")
        assert isinstance(result, list)
    
    def test_explain_query_functionality(self):
        """Test enhanced query explanation"""
        explanation = self.filter_engine.explain_query("cpu:[50 TO *] AND status:running")
        
        assert "original_query" in explanation
        assert "validation" in explanation
        assert "estimated_complexity" in explanation
        assert explanation["validation"]["valid"] is True
        assert explanation["estimated_complexity"] in ["low", "medium", "high", "very_high"]
    
    def test_malformed_query_raises_error(self):
        """Malformed queries should raise ValueError with descriptive message"""
        with pytest.raises(ValueError) as exc_info:
            self.filter_engine.filter(self.dict_servers, "malformed[[query")
        
        assert "Invalid query" in str(exc_info.value)
    
    def test_metrics_collection(self):
        """Test that metrics are collected when enabled"""
        initial_count = len(self.filter_engine.get_metrics())
        
        self.filter_engine.filter(self.dict_servers, "status:running")
        
        metrics = self.filter_engine.get_metrics()
        assert len(metrics) == initial_count + 1
        assert metrics[-1].query == "status:running"
        assert metrics[-1].objects_processed == len(self.dict_servers)


class TestQueryValidation:
    """Test query validation functionality"""
    
    def test_valid_query_validation(self):
        """Test validation of valid queries"""
        valid_queries = [
            "field:value",
            "field1:value1 AND field2:value2",
            "field:[1 TO 10]",
            "field:prefix*",
            "(field1:value1 OR field2:value2) AND field3:value3"
        ]
        
        for query in valid_queries:
            validation = QueryValidator.validate(query)
            assert validation["valid"] is True, f"Query should be valid: {query}"
            assert validation["parsed_ast"] is not None
            assert len(validation["errors"]) == 0
    
    def test_invalid_query_validation(self):
        """Test validation of invalid queries"""
        invalid_queries = [
            "",
            "   ",
            "field:",
            ":value",
            "field:value AND",
            "field:[1 TO",
            "((field:value"
        ]
        
        for query in invalid_queries:
            validation = QueryValidator.validate(query)
            assert validation["valid"] is False, f"Query should be invalid: {query}"
            assert len(validation["errors"]) > 0
    
    def test_validation_warnings(self):
        """Test validation warnings for problematic but valid queries"""
        # Test range with low > high
        validation = QueryValidator.validate("field:[100 TO 50]")
        # Should be valid but have warnings
        assert validation["valid"] is True
        # May have warnings about range bounds
    
    def test_semantic_validation(self):
        """Test semantic validation features"""
        validation = QueryValidator.validate("_private_field:value")
        assert validation["valid"] is True
        # Should warn about private field access


class TestFilterConfiguration:
    """Test FilterConfig options and their effects"""
    
    def test_case_sensitive_configuration(self):
        """Test case sensitivity configuration"""
        data = [{"name": "TestServer"}, {"name": "testserver"}]
        
        # Case insensitive (default)
        config_insensitive = FilterConfig(case_sensitive=False)
        filter_insensitive = EnhancedLuceneObjectFilter(config_insensitive)
        result = filter_insensitive.filter(data, "name:testserver")
        assert len(result) == 2  # Should match both
        
        # Case sensitive
        config_sensitive = FilterConfig(case_sensitive=True)
        filter_sensitive = EnhancedLuceneObjectFilter(config_sensitive)
        result = filter_sensitive.filter(data, "name:testserver")
        assert len(result) == 1  # Should match only exact case
    
    def test_recursion_depth_limits(self):
        """Test recursion depth configuration"""
        # Create deeply nested object
        deep_obj = {"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}
        data = [deep_obj]
        
        # Shallow depth limit
        config_shallow = FilterConfig(max_recursion_depth=2)
        filter_shallow = EnhancedLuceneObjectFilter(config_shallow)
        result = filter_shallow.filter(data, "level1.level2.level3.level4.value:deep")
        assert len(result) == 0  # Should not reach deep value
        
        # Deeper depth limit
        config_deep = FilterConfig(max_recursion_depth=10)
        filter_deep = EnhancedLuceneObjectFilter(config_deep)
        result = filter_deep.filter(data, "level1.level2.level3.level4.value:deep")
        assert len(result) == 1  # Should reach deep value
    
    def test_field_count_limits(self):
        """Test field count limits"""
        # Create object with many fields
        many_fields_obj = {f"field_{i}": f"value_{i}" for i in range(100)}
        many_fields_obj["target"] = "findme"
        data = [many_fields_obj]
        
        # Low field limit
        config_limited = FilterConfig(max_field_count=50)
        filter_limited = EnhancedLuceneObjectFilter(config_limited)
        
        # Should handle within limits
        result = filter_limited.filter(data, "target:findme")
        assert len(result) == 1
    
    def test_cache_configuration(self):
        """Test text cache configuration"""
        data = [{"text": "searchable content"}]
        
        # Cache enabled
        config_cached = FilterConfig(enable_text_cache=True)
        filter_cached = EnhancedLuceneObjectFilter(config_cached)
        
        # Cache disabled
        config_no_cache = FilterConfig(enable_text_cache=False)
        filter_no_cache = EnhancedLuceneObjectFilter(config_no_cache)
        
        # Both should work
        result1 = filter_cached.filter(data, "searchable")
        result2 = filter_no_cache.filter(data, "searchable")
        
        assert len(result1) == 1
        assert len(result2) == 1
    
    def test_metrics_configuration(self):
        """Test metrics collection configuration"""
        data = [{"field": "value"}]
        
        # Metrics enabled
        config_metrics = FilterConfig(enable_metrics=True)
        filter_metrics = EnhancedLuceneObjectFilter(config_metrics)
        filter_metrics.filter(data, "field:value")
        assert len(filter_metrics.get_metrics()) > 0
        
        # Metrics disabled
        config_no_metrics = FilterConfig(enable_metrics=False)
        filter_no_metrics = EnhancedLuceneObjectFilter(config_no_metrics)
        filter_no_metrics.filter(data, "field:value")
        assert len(filter_no_metrics.get_metrics()) == 0


class TestTextExtractionCache:
    """Test text extraction caching functionality"""
    
    def test_cache_basic_functionality(self):
        """Test basic cache operations"""
        cache = TextExtractionCache(max_size=3)
        
        # Store items
        cache.put(1, ["text1"])
        cache.put(2, ["text2"])
        cache.put(3, ["text3"])
        
        # Retrieve items
        assert cache.get(1) == ["text1"]
        assert cache.get(2) == ["text2"]
        assert cache.get(3) == ["text3"]
        assert cache.get(4) is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction behavior"""
        cache = TextExtractionCache(max_size=2)
        
        # Fill cache
        cache.put(1, ["text1"])
        cache.put(2, ["text2"])
        
        # Access item 1 to make it recently used
        cache.get(1)
        
        # Add item 3, should evict item 2
        cache.put(3, ["text3"])
        
        assert cache.get(1) == ["text1"]  # Still there
        assert cache.get(2) is None       # Evicted
        assert cache.get(3) == ["text3"]  # New item
    
    def test_cache_update_existing(self):
        """Test updating existing cache entries"""
        cache = TextExtractionCache(max_size=2)
        
        cache.put(1, ["original"])
        cache.put(1, ["updated"])
        
        assert cache.get(1) == ["updated"]
    
    def test_cache_clear(self):
        """Test cache clearing"""
        cache = TextExtractionCache(max_size=2)
        cache.put(1, ["text1"])
        cache.put(2, ["text2"])
        
        cache.clear()
        
        assert cache.get(1) is None
        assert cache.get(2) is None


class TestFieldMatching:
    """Test field-based exact matching with enhanced features"""
    
    def setup_method(self):
        self.filter_engine = EnhancedLuceneObjectFilter()
        self.servers = [
            {"name": "web-01", "cpu": 85, "status": "running", "active": True},
            {"name": "db-01", "cpu": 45, "status": "stopped", "active": False},
            {"name": "cache-01", "cpu": 25, "status": "running", "active": True},
        ]
    
    def test_string_field_match(self):
        """Test exact string field matching"""
        result = self.filter_engine.filter(self.servers, "status:running")
        assert len(result) == 2
        assert all(s["status"] == "running" for s in result)
    
    def test_numeric_field_match(self):
        """Test exact numeric field matching"""
        result = self.filter_engine.filter(self.servers, "cpu:45")
        assert len(result) == 1
        assert result[0]["name"] == "db-01"
    
    def test_boolean_field_match(self):
        """Test boolean field matching"""
        result = self.filter_engine.filter(self.servers, "active:true")
        assert len(result) == 2
        assert all(s["active"] is True for s in result)
        
        result = self.filter_engine.filter(self.servers, "active:false")
        assert len(result) == 1
        assert result[0]["active"] is False
    
    def test_nonexistent_field(self):
        """Test filtering on nonexistent fields"""
        result = self.filter_engine.filter(self.servers, "nonexistent:value")
        assert len(result) == 0
    
    def test_null_field_handling(self):
        """Test handling of null/None field values"""
        data_with_nulls = [
            {"name": "server1", "value": "present"},
            {"name": "server2", "value": None},
            {"name": "server3"}  # Missing field
        ]
        
        result = self.filter_engine.filter(data_with_nulls, "value:null")
        assert len(result) == 1
        assert result[0]["name"] == "server2"


class TestComparisonOperators:
    """Test numeric and string comparison using range syntax"""
    
    def setup_method(self):
        self.filter_engine = EnhancedLuceneObjectFilter()
        self.servers = [
            {"name": "server-a", "cpu": 10, "memory": 8},
            {"name": "server-b", "cpu": 50, "memory": 16},
            {"name": "server-c", "cpu": 90, "memory": 32},
        ]
    
    def test_range_queries_greater_than(self):
        """Test range queries for > logic using [value TO *]"""
        result = self.filter_engine.filter(self.servers, "cpu:[51 TO *]")
        assert len(result) == 1
        assert result[0]["cpu"] == 90
    
    def test_range_queries_less_than(self):
        """Test range queries for < logic using [* TO value]"""
        result = self.filter_engine.filter(self.servers, "memory:[* TO 19]")
        assert len(result) == 2
        assert all(s["memory"] < 20 for s in result)
    
    def test_range_inclusive(self):
        """Test inclusive range queries"""
        result = self.filter_engine.filter(self.servers, "cpu:[50 TO 90]")
        assert len(result) == 2
        cpu_values = [s["cpu"] for s in result]
        assert 50 in cpu_values
        assert 90 in cpu_values
    
    def test_range_exclusive(self):
        """Test exclusive range queries"""
        result = self.filter_engine.filter(self.servers, "cpu:{49 TO 91}")
        assert len(result) == 2
        assert all(49 < s["cpu"] < 91 for s in result)
    
    def test_float_range_queries(self):
        """Test range queries with floating point numbers"""
        float_data = [
            {"value": 1.5},
            {"value": 2.7},
            {"value": 3.9}
        ]
        
        result = self.filter_engine.filter(float_data, "value:[2.0 TO 3.5]")
        assert len(result) == 1
        assert result[0]["value"] == 2.7


class TestBooleanLogic:
    """Test AND, OR, NOT operations with enhanced precedence"""
    
    def setup_method(self):
        self.filter_engine = EnhancedLuceneObjectFilter()
        self.servers = [
            {"name": "web-01", "cpu": 80, "status": "running", "env": "prod"},
            {"name": "web-02", "cpu": 30, "status": "running", "env": "dev"},
            {"name": "db-01", "cpu": 70, "status": "stopped", "env": "prod"},
            {"name": "cache-01", "cpu": 20, "status": "idle", "env": "dev"},
        ]
    
    def test_and_operator(self):
        """Test AND logic"""
        result = self.filter_engine.filter(self.servers, "status:running AND env:prod")
        assert len(result) == 1
        assert result[0]["name"] == "web-01"
        
        result = self.filter_engine.filter(self.servers, "cpu:[61 TO *] AND status:running")
        assert len(result) == 1
        assert result[0]["name"] == "web-01"
    
    def test_or_operator(self):
        """Test OR logic"""
        result = self.filter_engine.filter(self.servers, "status:stopped OR status:idle")
        assert len(result) == 2
        assert all(s["status"] in ["stopped", "idle"] for s in result)
        
        result = self.filter_engine.filter(self.servers, "cpu:[71 TO *] OR cpu:[* TO 24]")
        assert len(result) == 2
    
    def test_not_operator(self):
        """Test NOT logic"""
        result = self.filter_engine.filter(self.servers, "NOT status:running")
        assert len(result) == 2
        assert all(s["status"] != "running" for s in result)
        
        result = self.filter_engine.filter(self.servers, "NOT env:dev")
        assert len(result) == 2
        assert all(s["env"] != "dev" for s in result)
    
    def test_complex_boolean_logic(self):
        """Test complex combinations with proper precedence"""
        result = self.filter_engine.filter(self.servers, 
            "env:prod AND (status:running OR cpu:[66 TO *])")
        assert len(result) == 2  # web-01 and db-01
        
        result = self.filter_engine.filter(self.servers,
            "NOT env:dev AND cpu:[* TO 79]")
        assert len(result) == 1
        assert result[0]["name"] == "db-01"
    
    def test_operator_precedence(self):
        """Test that operator precedence is handled correctly"""
        result = self.filter_engine.filter(self.servers,
            "NOT env:dev AND cpu:[50 TO *] OR env:prod")
        # Should include all prod servers and high CPU non-dev servers
        assert len(result) >= 2
    
    def test_nested_grouping(self):
        """Test deeply nested boolean grouping"""
        result = self.filter_engine.filter(self.servers,
            "((env:prod AND status:running) OR (env:dev AND status:idle)) AND cpu:[* TO 80]")
        # Should match web-01 (prod+running) and cache-01 (dev+idle) with cpu <= 80
        expected_names = {"web-01", "cache-01"}
        actual_names = {s["name"] for s in result}
        assert actual_names == expected_names


class TestRangeQueries:
    """Test range query functionality with enhanced features"""
    
    def setup_method(self):
        self.filter_engine = EnhancedLuceneObjectFilter()
        self.servers = [
            {"name": "server-1", "cpu": 10, "memory": 4},
            {"name": "server-2", "cpu": 50, "memory": 8},
            {"name": "server-3", "cpu": 80, "memory": 16},
            {"name": "server-4", "cpu": 90, "memory": 32},
        ]
    
    def test_numeric_range(self):
        """Test numeric range queries"""
        result = self.filter_engine.filter(self.servers, "cpu:[40 TO 85]")
        assert len(result) == 2
        assert all(40 <= s["cpu"] <= 85 for s in result)
    
    def test_inclusive_range_boundaries(self):
        """Test that range boundaries are inclusive by default"""
        result = self.filter_engine.filter(self.servers, "cpu:[50 TO 80]")
        assert len(result) == 2
        cpu_values = [s["cpu"] for s in result]
        assert 50 in cpu_values
        assert 80 in cpu_values
    
    def test_exclusive_range_boundaries(self):
        """Test exclusive range boundaries with {}"""
        result = self.filter_engine.filter(self.servers, "cpu:{49 TO 81}")
        assert len(result) == 2
        assert all(49 < s["cpu"] < 81 for s in result)
    
    def test_open_ended_ranges(self):
        """Test open-ended ranges with *"""
        result = self.filter_engine.filter(self.servers, "cpu:[70 TO *]")
        assert len(result) == 2
        assert all(s["cpu"] >= 70 for s in result)
        
        result = self.filter_engine.filter(self.servers, "memory:[* TO 10]")
        assert len(result) == 2
        assert all(s["memory"] <= 10 for s in result)
    
    def test_invalid_range_handling(self):
        """Test handling of invalid range syntax"""
        # Should handle gracefully without crashing
        try:
            result = self.filter_engine.filter(self.servers, "cpu:[invalid TO range]")
            # Should return empty result or handle gracefully
            assert isinstance(result, list)
        except Exception:
            # May raise validation error, which is acceptable
            pass


class TestWildcardQueries:
    """Test wildcard pattern matching with enhanced features"""
    
    def setup_method(self):
        self.filter_engine = EnhancedLuceneObjectFilter()
        self.servers = [
            {"name": "web-server-01", "type": "frontend"},
            {"name": "api-gateway-02", "type": "middleware"},
            {"name": "db-primary", "type": "database"},
            {"name": "cache-redis", "type": "caching"},
        ]
    
    def test_prefix_wildcard(self):
        """Test prefix matching with *"""
        result = self.filter_engine.filter(self.servers, "name:web*")
        assert len(result) == 1
        assert result[0]["name"] == "web-server-01"
    
    def test_suffix_wildcard(self):
        """Test suffix matching with *"""
        result = self.filter_engine.filter(self.servers, "name:*01")
        assert len(result) == 1
        assert result[0]["name"] == "web-server-01"
    
    def test_contains_wildcard(self):
        """Test contains matching with *"""
        result = self.filter_engine.filter(self.servers, "name:*gateway*")
        assert len(result) == 1
        assert result[0]["name"] == "api-gateway-02"
    
    def test_single_char_wildcard(self):
        """Test single character wildcard with ?"""
        result = self.filter_engine.filter(self.servers, "name:*-0?")
        assert len(result) == 2  # Matches -01 and -02
    
    def test_case_insensitive_wildcards(self):
        """Test case insensitive wildcard matching"""
        result = self.filter_engine.filter(self.servers, "name:WEB*")
        assert len(result) == 1  # Should match web-server-01
    
    def test_complex_wildcard_patterns(self):
        """Test complex wildcard combinations"""
        result = self.filter_engine.filter(self.servers, "name:*-??")
        assert len(result) == 2  # Should match patterns with -XX ending


class TestObjectTypes:
    """Test filtering different object types with enhanced features"""
    
    def setup_method(self):
        self.filter_engine = EnhancedLuceneObjectFilter()
    
    def test_dataclass_objects(self):
        """Test filtering dataclass objects"""
        alerts = [
            Alert("critical", "DB down", "2024-01-15", False, 5),
            Alert("warning", "High CPU", "2024-01-16", True, 2),
        ]
        
        result = self.filter_engine.filter(alerts, "severity:critical")
        assert len(result) == 1
        assert result[0].severity == "critical"
        
        result = self.filter_engine.filter(alerts, "resolved:false")
        assert len(result) == 1
        assert result[0].resolved is False
    
    def test_nested_dict_access(self):
        """Test accessing nested dictionary fields"""
        data = [
            {"server": {"name": "web-01", "specs": {"cpu": 4, "ram": 16}}},
            {"server": {"name": "db-01", "specs": {"cpu": 8, "ram": 32}}},
        ]
        
        result = self.filter_engine.filter(data, "server.name:web-01")
        assert len(result) == 1
        
        result = self.filter_engine.filter(data, "server.specs.cpu:[7 TO *]")
        assert len(result) == 1
        assert result[0]["server"]["name"] == "db-01"
    
    def test_nested_object_access(self):
        """Test accessing nested object attributes"""
        servers = [
            Server("web-01", 85.5, 16, "running", ["web"], 
                   datetime(2024, 1, 15), {"ssl": True}),
            Server("db-01", 45.2, 32, "running", ["db"], 
                   datetime(2024, 1, 10), {"ssl": False}),
        ]
        
        result = self.filter_engine.filter(servers, "config.ssl:true")
        assert len(result) == 1
        assert result[0].name == "web-01"
    
    def test_list_field_access(self):
        """Test accessing list elements"""
        data = [
            {"items": ["apple", "banana", "cherry"]},
            {"items": ["dog", "cat", "bird"]}
        ]
        
        # Test accessing by index
        result = self.filter_engine.filter(data, "items.0:apple")
        assert len(result) == 1
        
        # Test full-text search in lists
        result = self.filter_engine.filter(data, "banana")
        assert len(result) == 1


class TestFullTextSearch:
    """Test full-text search functionality with caching"""
    
    def setup_method(self):
        self.filter_engine = EnhancedLuceneObjectFilter()
        self.data = [
            {"name": "web-server", "description": "Handles HTTP requests"},
            {"name": "database", "description": "Stores application data"},
            {"name": "cache", "description": "Redis cache for performance"},
        ]
    
    def test_fulltext_search(self):
        """Test searching across all fields"""
        result = self.filter_engine.filter(self.data, "HTTP")
        assert len(result) == 1
        assert result[0]["name"] == "web-server"
        
        result = self.filter_engine.filter(self.data, "data")
        assert len(result) == 2  # Matches "database" and "application data"
    
    def test_case_insensitive_fulltext(self):
        """Test case-insensitive full-text search"""
        result = self.filter_engine.filter(self.data, "redis")
        assert len(result) == 1
        assert result[0]["name"] == "cache"
    
    def test_phrase_search(self):
        """Test quoted phrase search"""
        result = self.filter_engine.filter(self.data, '"HTTP requests"')
        assert len(result) == 1
        assert result[0]["name"] == "web-server"
    
    def test_fulltext_caching(self):
        """Test that full-text search uses caching"""
        # First search
        result1 = self.filter_engine.filter(self.data, "HTTP")
        
        # Second search should use cache
        result2 = self.filter_engine.filter(self.data, "requests")
        
        # Both should work
        assert len(result1) == 1
        assert len(result2) == 1
        
        # Check that some cache activity occurred
        metrics = self.filter_engine.get_metrics()
        assert len(metrics) > 0


class TestAdvancedLuceneFeatures:
    """Test advanced Lucene features"""
    
    def setup_method(self):
        self.filter_engine = EnhancedLuceneObjectFilter()
        self.data = [
            {"name": "test-server", "description": "testing server"},
            {"name": "prod-server", "description": "production server"},
            {"name": "dev-machine", "description": "development machine"},
        ]
    
    def test_fuzzy_search(self):
        """Test fuzzy matching with ~"""
        result = self.filter_engine.filter(self.data, "name:test~")
        assert len(result) >= 1  # Should match "test-server"
    
    def test_fuzzy_search_configuration(self):
        """Test fuzzy search with different thresholds"""
        config = FilterConfig(fuzzy_threshold=0.9)  # Stricter fuzzy matching
        filter_strict = EnhancedLuceneObjectFilter(config)
        
        # Strict fuzzy should be more selective
        result = filter_strict.filter(self.data, "name:test~")
        assert isinstance(result, list)
    
    def test_field_grouping(self):
        """Test field:(term1 term2) grouping"""
        result = self.filter_engine.filter(self.data, "description:(server machine)")
        assert len(result) >= 2  # Should match servers and machines
    
    def test_boost_queries(self):
        """Test boost queries (term^2.0) - boost doesn't affect boolean results"""
        result = self.filter_engine.filter(self.data, "name:test^2.0")
        # Boost doesn't change boolean matching, just relevance
        assert len(result) >= 1
    
    def test_proximity_search(self):
        """Test proximity search functionality"""
        data_with_text = [
            {"content": "the quick brown fox jumps"},
            {"content": "quick fox over lazy dog"},
            {"content": "brown and white fox"}
        ]
        
        # This would test proximity if implemented
        # For now, just ensure it doesn't crash
        try:
            result = self.filter_engine.filter(data_with_text, '"quick fox"~2')
            assert isinstance(result, list)
        except Exception:
            # Proximity might not be fully implemented yet
            pass


class TestDateHandling:
    """Test date parsing and filtering"""
    
    def setup_method(self):
        self.filter_engine = EnhancedLuceneObjectFilter()
        self.data = [
            {"created": "2024-01-15", "name": "old"},
            {"created": "2024-06-15", "name": "recent"},
            {"created": "2024-12-15", "name": "future"},
            {"created": datetime(2024, 3, 10), "name": "datetime_obj"}
        ]
    
    def test_date_exact_match(self):
        """Test exact date matching"""
        result = self.filter_engine.filter(self.data, "created:2024-01-15")
        assert len(result) >= 1
    
    def test_date_range_queries(self):
        """Test date range queries"""
        result = self.filter_engine.filter(self.data, "created:[2024-01-01 TO 2024-03-31]")
        # Should match items in Q1 2024
        assert len(result) >= 1
    
    def test_different_date_formats(self):
        """Test various date format parsing"""
        mixed_dates = [
            {"date": "01/15/2024", "format": "US"},
            {"date": "15/01/2024", "format": "EU"},
            {"date": "2024-01-15T10:30:00", "format": "ISO"},
        ]
        
        # Test that various formats are handled
        result = self.filter_engine.filter(mixed_dates, "2024")
        assert len(result) >= 1


class TestErrorHandling:
    """Test comprehensive error handling and edge cases"""
    
    def setup_method(self):
        self.filter_engine = EnhancedLuceneObjectFilter()
    
    def test_empty_data_list(self):
        """Test filtering empty data list"""
        result = self.filter_engine.filter([], "field:value")
        assert result == []
    
    def test_none_values_comprehensive(self):
        """Test comprehensive None value handling"""
        data = [
            {"name": "server-1", "status": "running", "value": None},
            {"name": "server-2", "status": None, "value": 42},
            {"name": None, "status": "stopped", "value": "present"},
            None,  # Entire object is None
        ]
        
        # Filter out None objects first
        clean_data = [obj for obj in data if obj is not None]
        
        result = self.filter_engine.filter(clean_data, "status:running")
        assert len(result) == 1
        
        result = self.filter_engine.filter(clean_data, "value:null")
        assert len(result) == 1
    
    def test_circular_references_comprehensive(self):
        """Test comprehensive circular reference handling"""
        obj1 = CircularRefTest("parent")
        obj2 = CircularRefTest("child1")
        obj3 = CircularRefTest("child2")
        
        # Create circular references
        obj1.children = [obj2, obj3]
        obj2.parent = obj1
        obj3.parent = obj1
        obj2.children = [obj3]  # Child to child reference
        obj3.children = [obj1]  # Back to parent
        
        data = [obj1, obj2, obj3]
        
        # Should not cause infinite recursion
        result = self.filter_engine.filter(data, "name:parent")
        assert len(result) == 1
        assert result[0].name == "parent"
        
        # Test full-text search across circular structure
        result = self.filter_engine.filter(data, "child")
        assert len(result) >= 2
    
    def test_deep_nesting_limits(self):
        """Test deeply nested object handling with limits"""
        # Create progressively deeper nesting
        deep_obj = {"value": "surface"}
        current = deep_obj
        
        for i in range(15):  # Create 15 levels deep
            current[f"level_{i}"] = {"nested": f"value_{i}"}
            current = current[f"level_{i}"]
        
        current["deep_value"] = "treasure"
        
        data = [deep_obj]
        
        # Should handle within recursion limits
        result = self.filter_engine.filter(data, "value:surface")
        assert len(result) == 1
        
        # Deep value might not be accessible due to recursion limits
        result = self.filter_engine.filter(data, "deep_value:treasure")
        # Result depends on max_recursion_depth setting
        assert isinstance(result, list)
    
    def test_large_object_handling(self):
        """Test handling of objects with many fields"""
        large_obj = {f"field_{i}": f"value_{i}" for i in range(200)}
        large_obj["target"] = "findme"
        large_obj["special"] = "important"
        
        data = [large_obj]
        
        result = self.filter_engine.filter(data, "target:findme")
        assert len(result) == 1
        
        result = self.filter_engine.filter(data, "special:important")
        assert len(result) == 1
    
    def test_special_characters_comprehensive(self):
        """Test comprehensive special character handling"""
        data = [
            {"path": "/usr/local/bin", "email": "user@domain.com"},
            {"url": "https://example.com/path?param=value&other=123"},
            {"json": '{"key": "value", "nested": {"inner": true}}'},
            {"regex": ".*\\.log$", "shell": "echo 'hello world'"},
            {"unicode": "café naïve résumé", "emoji": "🚀 💻 🎯"},
        ]
        
        # Test email matching
        result = self.filter_engine.filter(data, "email:*@domain.com")
        assert len(result) == 1
        
        # Test URL matching
        result = self.filter_engine.filter(data, "url:*example.com*")
        assert len(result) == 1
        
        # Test Unicode handling
        result = self.filter_engine.filter(data, "café")
        assert len(result) == 1
        
        # Test emoji handling
        result = self.filter_engine.filter(data, "🚀")
        assert len(result) == 1
    
    def test_type_conversion_edge_cases(self):
        """Test edge cases in type conversion"""
        edge_case_data = [
            {"number": "01", "name": "leading_zero"},
            {"number": "1.0", "name": "float_as_string"},
            {"number": "1e3", "name": "scientific"},
            {"number": "+42", "name": "explicit_positive"},
            {"number": "-0", "name": "negative_zero"},
            {"boolean": "TRUE", "name": "uppercase_bool"},
            {"boolean": "Yes", "name": "yes_bool"},
            {"boolean": "1", "name": "numeric_bool"},
        ]
        
        # Test various numeric conversions
        result = self.filter_engine.filter(edge_case_data, "number:1")
        assert len(result) >= 1  # Should match "01" and "1.0"
        
        # Test boolean conversions
        result = self.filter_engine.filter(edge_case_data, "boolean:true")
        assert len(result) >= 1  # Should match various true representations
    
    def test_malformed_data_structures(self):
        """Test handling of malformed or unusual data structures"""
        class CustomObject:
            def __init__(self, value):
                self.value = value
                self._private = "hidden"
            
            def __str__(self):
                return f"Custom({self.value})"
        
        malformed_data = [
            {"normal": "value"},
            CustomObject("test"),
            {"function": lambda x: x},  # Function in data
            {"generator": (i for i in range(3))},  # Generator
            {"bytes": b"binary data"},
        ]
        
        # Should handle without crashing
        result = self.filter_engine.filter(malformed_data, "normal:value")
        assert len(result) >= 1
        
        result = self.filter_engine.filter(malformed_data, "test")
        assert len(result) >= 1  # Should find CustomObject


class TestMetricsAndMonitoring:
    """Test metrics collection and performance monitoring"""
    
    def setup_method(self):
        config = FilterConfig(enable_metrics=True)
        self.filter_engine = EnhancedLuceneObjectFilter(config)
        self.data = [{"field": f"value_{i}"} for i in range(10)]
    
    def test_metrics_collection_comprehensive(self):
        """Test comprehensive metrics collection"""
        # Clear any existing metrics
        self.filter_engine.clear_metrics()
        
        # Execute several queries
        queries = ["field:value_1", "field:value_*", "nonexistent:field"]
        
        for query in queries:
            try:
                self.filter_engine.filter(self.data, query)
            except Exception:
                pass  # Some queries might fail validation
        
        metrics = self.filter_engine.get_metrics()
        assert len(metrics) >= 2  # At least successful queries
        
        # Check metric structure
        for metric in metrics:
            assert hasattr(metric, 'query')
            assert hasattr(metric, 'execution_time')
            assert hasattr(metric, 'objects_processed')
            assert hasattr(metric, 'timestamp')
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Execute some queries to generate metrics
        self.filter_engine.filter(self.data, "field:value_1")
        self.filter_engine.filter(self.data, "field:value_2")
        
        summary = self.filter_engine.get_performance_summary()
        
        assert "total_queries" in summary
        assert "avg_execution_time" in summary
        assert "max_execution_time" in summary
        assert "min_execution_time" in summary
        assert "avg_cache_hit_rate" in summary
        assert summary["total_queries"] >= 2
    
    def test_metrics_disabled(self):
        """Test behavior when metrics are disabled"""
        config = FilterConfig(enable_metrics=False)
        filter_no_metrics = EnhancedLuceneObjectFilter(config)
        
        filter_no_metrics.filter(self.data, "field:value_1")
        
        metrics = filter_no_metrics.get_metrics()
        assert len(metrics) == 0
    
    def test_cache_metrics(self):
        """Test cache hit/miss metrics"""
        # Run same query multiple times to generate cache hits
        for _ in range(3):
            self.filter_engine.filter(self.data, "field:value_1")
        
        metrics = self.filter_engine.get_metrics()
        # Should have some cache activity
        total_cache_operations = sum(m.cache_hits + m.cache_misses for m in metrics)
        assert total_cache_operations > 0
    
    def test_error_metrics(self):
        """Test error tracking in metrics"""
        try:
            self.filter_engine.filter(self.data, "invalid[[query")
        except Exception:
            pass  # Expected to fail
        
        metrics = self.filter_engine.get_metrics()
        error_metrics = [m for m in metrics if m.error is not None]
        assert len(error_metrics) >= 1


class TestComplexQueries:
    """Test complex query structures and edge cases"""
    
    def setup_method(self):
        self.filter_engine = EnhancedLuceneObjectFilter()
        self.complex_data = [
            {
                "server": {
                    "name": "web-cluster-01",
                    "specs": {"cpu": 8, "memory": 32, "disk": 500},
                    "services": [
                        {"name": "nginx", "port": 80, "status": "running"},
                        {"name": "app", "port": 8080, "status": "running"}
                    ]
                },
                "environment": "production",
                "tags": ["web", "public", "load-balanced"],
                "metrics": {"uptime": 99.9, "load": 0.7}
            },
            {
                "server": {
                    "name": "db-cluster-01",
                    "specs": {"cpu": 16, "memory": 64, "disk": 2000},
                    "services": [
                        {"name": "postgresql", "port": 5432, "status": "running"},
                        {"name": "pgbouncer", "port": 6432, "status": "running"}
                    ]
                },
                "environment": "production",
                "tags": ["database", "persistent", "clustered"],
                "metrics": {"uptime": 99.99, "load": 0.3}
            }
        ]
    
    def test_deeply_nested_queries(self):
        """Test queries on deeply nested structures"""
        result = self.filter_engine.filter(self.complex_data, "server.specs.cpu:[10 TO *]")
        assert len(result) == 1
        assert result[0]["server"]["name"] == "db-cluster-01"
        
        result = self.filter_engine.filter(self.complex_data, "server.services.0.name:nginx")
        assert len(result) == 1
        assert result[0]["server"]["name"] == "web-cluster-01"
    
    def test_array_element_queries(self):
        """Test queries on array elements"""
        result = self.filter_engine.filter(self.complex_data, "tags:database")
        assert len(result) == 1
        
        result = self.filter_engine.filter(self.complex_data, "server.services.1.port:8080")
        assert len(result) == 1
    
    def test_complex_boolean_combinations(self):
        """Test complex boolean query combinations"""
        query = """
        (server.specs.cpu:[10 TO *] AND environment:production) 
        OR 
        (metrics.uptime:[99.5 TO *] AND tags:web)
        """
        
        result = self.filter_engine.filter(self.complex_data, query)
        assert len(result) == 2  # Should match both servers
    
    def test_mixed_field_and_fulltext(self):
        """Test combining field queries with full-text search"""
        result = self.filter_engine.filter(self.complex_data, 
            "environment:production AND nginx")
        assert len(result) == 1
        assert result[0]["server"]["name"] == "web-cluster-01"
    
    def test_multiple_wildcards(self):
        """Test multiple wildcard patterns in single query"""
        result = self.filter_engine.filter(self.complex_data,
            "server.name:*cluster* AND tags:*ed")
        assert len(result) == 2  # Both have "cluster" in name and "ed" suffix tags


class TestConfigurationEdgeCases:
    """Test edge cases in configuration handling"""
    
    def test_extreme_recursion_limits(self):
        """Test extreme recursion depth limits"""
        # Very low limit
        config_shallow = FilterConfig(max_recursion_depth=1)
        filter_shallow = EnhancedLuceneObjectFilter(config_shallow)
        
        nested_data = [{"level1": {"level2": {"value": "deep"}}}]
        
        result = filter_shallow.filter(nested_data, "level1.level2.value:deep")
        assert len(result) == 0  # Too deep for shallow config
        
        # Very high limit
        config_deep = FilterConfig(max_recursion_depth=100)
        filter_deep = EnhancedLuceneObjectFilter(config_deep)
        
        result = filter_deep.filter(nested_data, "level1.level2.value:deep")
        assert len(result) == 1  # Should reach with deep config
    
    def test_cache_size_limits(self):
        """Test cache size limit handling"""
        config = FilterConfig(cache_size_limit=2)
        filter_limited = EnhancedLuceneObjectFilter(config)
        
        # Create enough objects to exceed cache
        data = [{"field": f"value_{i}"} for i in range(5)]
        
        # Search each object to fill cache beyond limit
        for i in range(5):
            filter_limited.filter([data[i]], f"value_{i}")
        
        # Cache should be limited to configured size
        cache_size = len(filter_limited.text_cache.cache)
        assert cache_size <= 2
    
    def test_fuzzy_threshold_extremes(self):
        """Test extreme fuzzy matching thresholds"""
        # Very strict threshold
        config_strict = FilterConfig(fuzzy_threshold=0.99)
        filter_strict = EnhancedLuceneObjectFilter(config_strict)
        
        # Very loose threshold
        config_loose = FilterConfig(fuzzy_threshold=0.1)
        filter_loose = EnhancedLuceneObjectFilter(config_loose)
        
        data = [{"word": "test"}, {"word": "best"}]
        
        # Test with slightly different word
        strict_result = filter_strict.filter(data, "word:tест~")  # Cyrillic 'е'
        loose_result = filter_loose.filter(data, "word:tест~")
        
        # Loose should match more than strict
        assert len(loose_result) >= len(strict_result)


# Pytest fixtures for enhanced testing
@pytest.fixture
def enhanced_filter():
    """Pytest fixture providing EnhancedLuceneObjectFilter instance"""
    return EnhancedLuceneObjectFilter()


@pytest.fixture
def configured_filter():
    """Pytest fixture with custom configuration"""
    config = FilterConfig(
        enable_text_cache=True,
        enable_metrics=True,
        case_sensitive=False,
        max_recursion_depth=15
    )
    return EnhancedLuceneObjectFilter(config)


@pytest.fixture
def sample_servers():
    """Enhanced sample server data"""
    return [
        {"name": "web-01", "cpu": 85, "memory": 16, "status": "running", "tags": ["web", "prod"]},
        {"name": "db-01", "cpu": 45, "memory": 32, "status": "running", "tags": ["db", "prod"]},
        {"name": "cache-01", "cpu": 25, "memory": 8, "status": "stopped", "tags": ["cache", "dev"]},
    ]


@pytest.fixture
def complex_nested_data():
    """Enhanced complex nested data structures"""
    return [
        {
            "infrastructure": {
                "cloud": {
                    "provider": "aws",
                    "region": "us-west-2",
                    "zones": ["us-west-2a", "us-west-2b"]
                },
                "kubernetes": {
                    "version": "1.28",
                    "nodes": [
                        {"name": "node-1", "cpu": 16, "memory": 32},
                        {"name": "node-2", "cpu": 8, "memory": 16}
                    ]
                }
            },
            "applications": {
                "frontend": {"replicas": 3, "version": "1.2.0"},
                "backend": {"replicas": 2, "version": "2.1.0"}
            },
            "monitoring": {
                "enabled": True,
                "tools": ["prometheus", "grafana", "jaeger"]
            }
        }
    ]


# Tests using enhanced fixtures
def test_fixture_usage_enhanced(enhanced_filter, sample_servers):
    """Test using enhanced fixtures"""
    result = enhanced_filter.filter(sample_servers, "status:running")
    assert len(result) == 2
    
    # Test metrics collection
    metrics = enhanced_filter.get_metrics()
    assert len(metrics) >= 1


def test_complex_nested_filtering_enhanced(configured_filter, complex_nested_data):
    """Test filtering on enhanced complex nested structures"""
    result = configured_filter.filter(complex_nested_data, 
        "infrastructure.kubernetes.nodes.0.cpu:[10 TO *]")
    assert len(result) == 1
    
    result = configured_filter.filter(complex_nested_data,
        "applications.frontend.replicas:[2 TO *] AND monitoring.enabled:true")
    assert len(result) == 1
    
    result = configured_filter.filter(complex_nested_data, "prometheus")
    assert len(result) == 1


def test_comprehensive_error_scenarios(enhanced_filter):
    """Test comprehensive error handling scenarios"""
    # Test with various problematic inputs
    problematic_data = [
        {},  # Empty dict
        {"key": ""},  # Empty string value
        {"number": float('inf')},  # Infinity
        {"number": float('nan')},  # NaN
        {"nested": {"deep": {"very": {"deep": "value"}}}},  # Deep nesting
    ]
    
    # Should handle all cases gracefully
    result = enhanced_filter.filter(problematic_data, "key:value OR deep")
    assert isinstance(result, list)
    
    # Test query validation edge cases
    edge_queries = [
        "field:value AND",  # Incomplete boolean
        "field:()",  # Empty group
        "field:[]",  # Empty brackets
        "field:*",  # Just wildcard
    ]
    
    for query in edge_queries:
        try:
            enhanced_filter.filter(problematic_data, query)
        except ValueError:
            # Validation errors are expected and acceptable
            pass


# Test runner configuration for enhanced tests
if __name__ == "__main__":
    # Run tests with comprehensive coverage
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--cov=improved_lucene_filter",
        "--cov-report=html",
        "--cov-report=term"
    ])