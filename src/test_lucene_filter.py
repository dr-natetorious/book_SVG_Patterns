import pytest
from datetime import datetime, date
from dataclasses import dataclass
from typing import List, Dict, Any
import sys
import os

# Import the luqum-based implementation
from luqum_object_filter import LuceneObjectFilter, ObjectFilterVisitor
from luqum.parser import parser


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


class TestLuqumObjectFilterBasics:
    """Test basic functionality and setup"""
    
    def setup_method(self):
        """Setup test data for each test"""
        self.filter_engine = LuceneObjectFilter()
        
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
        result = self.filter_engine.filter(self.dict_servers, "cpu:>50")
        assert isinstance(result, list)
    
    def test_explain_query_functionality(self):
        """Test query explanation for debugging"""
        explanation = self.filter_engine.explain_query("cpu:>50 AND status:running")
        assert "AndOperation" in explanation or "AND" in explanation
    
    def test_malformed_query_graceful_degradation(self):
        """Malformed queries should not crash, return original data"""
        result = self.filter_engine.filter(self.dict_servers, "malformed[[query")
        assert isinstance(result, list)
        # Should either return all data or empty list, not crash


class TestFieldMatching:
    """Test field-based exact matching"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
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


class TestComparisonOperators:
    """Test numeric and string comparison operators - luqum handles these via range syntax"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
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


class TestBooleanLogic:
    """Test AND, OR, NOT operations"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
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
        """Test that operator precedence is handled correctly (NOT > AND > OR)"""
        # This should be parsed as: (NOT status:dev) AND (cpu:[50 TO *] OR env:prod)
        result = self.filter_engine.filter(self.servers,
            "NOT env:dev AND cpu:[50 TO *] OR env:prod")
        # Should include all prod servers and high CPU non-dev servers
        assert len(result) >= 2


class TestRangeQueries:
    """Test range query functionality"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
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


class TestWildcardQueries:
    """Test wildcard pattern matching"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
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


class TestObjectTypes:
    """Test filtering different object types"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
    
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


class TestFullTextSearch:
    """Test full-text search functionality"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
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


class TestAdvancedLuceneFeatures:
    """Test advanced Lucene features supported by luqum"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
        self.data = [
            {"name": "test-server", "description": "testing server"},
            {"name": "prod-server", "description": "production server"},
            {"name": "dev-machine", "description": "development machine"},
        ]
    
    def test_fuzzy_search(self):
        """Test fuzzy matching with ~"""
        result = self.filter_engine.filter(self.data, "name:test~")
        assert len(result) >= 1  # Should match "test-server"
    
    def test_field_grouping(self):
        """Test field:(term1 term2) grouping"""
        result = self.filter_engine.filter(self.data, "description:(server machine)")
        assert len(result) >= 2  # Should match servers and machines
    
    def test_escaped_characters(self):
        """Test handling of escaped characters"""
        data_with_special = [
            {"name": "test:server", "desc": "Has colon"},
            {"name": "normal-server", "desc": "Normal name"},
        ]
        result = self.filter_engine.filter(data_with_special, r"name:test\:server")
        # Note: Exact behavior depends on luqum's escaping implementation


class TestEdgeCasesAndSafety:
    """Test edge cases, error handling, and safety features"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
    
    def test_empty_data_list(self):
        """Test filtering empty data list"""
        result = self.filter_engine.filter([], "field:value")
        assert result == []
    
    def test_none_values(self):
        """Test handling None values in data"""
        data = [
            {"name": "server-1", "status": "running"},
            {"name": "server-2", "status": None},
            {"name": None, "status": "stopped"},
        ]
        
        result = self.filter_engine.filter(data, "status:running")
        assert len(result) == 1
        
        result = self.filter_engine.filter(data, "name:server-1")
        assert len(result) == 1
    
    def test_circular_references(self):
        """Test handling of circular object references"""
        obj1 = CircularRefTest("parent")
        obj2 = CircularRefTest("child")
        obj1.children = [obj2]
        obj2.parent = obj1
        
        data = [obj1, obj2]
        
        # Should not cause infinite recursion
        result = self.filter_engine.filter(data, "name:parent")
        assert len(result) == 1
        assert result[0].name == "parent"
    
    def test_deep_nesting(self):
        """Test deeply nested object structures"""
        deep_obj = {"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}
        data = [deep_obj, {"simple": "value"}]
        
        result = self.filter_engine.filter(data, "level1.level2.level3.level4.value:deep")
        assert len(result) == 1
    
    def test_large_object_handling(self):
        """Test handling of objects with many fields"""
        large_obj = {f"field_{i}": f"value_{i}" for i in range(1000)}
        large_obj["target"] = "findme"
        data = [large_obj]
        
        result = self.filter_engine.filter(data, "target:findme")
        assert len(result) == 1
    
    def test_special_characters_in_values(self):
        """Test handling special characters"""
        data = [
            {"path": "/usr/local/bin", "email": "user@domain.com"},
            {"url": "https://example.com/path?param=value"},
            {"json": '{"key": "value"}'},
        ]
        
        result = self.filter_engine.filter(data, "email:*@domain.com")
        assert len(result) == 1
        
        result = self.filter_engine.filter(data, "url:*example.com*")
        assert len(result) == 1


class TestPerformanceAndScalability:
    """Test performance considerations and scalability"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
    
    def test_large_dataset_filtering(self):
        """Test filtering performance with larger datasets"""
        # Generate test data
        large_data = []
        for i in range(1000):
            large_data.append({
                "id": i,
                "name": f"server-{i:04d}",
                "cpu": i % 100,
                "status": "running" if i % 2 == 0 else "stopped"
            })
        
        # Test basic filtering
        result = self.filter_engine.filter(large_data, "cpu:[51 TO *] AND status:running")
        assert len(result) > 0
        assert all(s["cpu"] > 50 and s["status"] == "running" for s in result)
        
        # Test wildcard on large dataset
        result = self.filter_engine.filter(large_data, "name:*-05*")
        assert len(result) > 0
    
    def test_complex_query_performance(self):
        """Test performance with complex nested queries"""
        data = [{"id": i, "group": i % 10, "active": i % 3 == 0} for i in range(100)]
        
        complex_query = (
            "(group:[0 TO 2] OR group:[7 TO 9]) AND "
            "(active:true OR id:[50 TO *]) AND "
            "NOT group:5"
        )
        
        result = self.filter_engine.filter(data, complex_query)
        assert isinstance(result, list)


class TestTypeConversion:
    """Test automatic type conversion and comparison"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
    
    def test_string_to_numeric_comparison(self):
        """Test string to numeric conversion in queries"""
        data = [{"count": 10}, {"count": 20}, {"count": 30}]
        result = self.filter_engine.filter(data, "count:20")
        assert len(result) == 1
        assert result[0]["count"] == 20
    
    def test_boolean_string_conversion(self):
        """Test boolean conversion from strings"""
        data = [{"enabled": True}, {"enabled": False}, {"enabled": "yes"}]
        result = self.filter_engine.filter(data, "enabled:true")
        assert len(result) == 1
        assert result[0]["enabled"] is True
    
    def test_mixed_type_ranges(self):
        """Test range queries with mixed types"""
        data = [
            {"value": 1.5}, 
            {"value": 2}, 
            {"value": "3.5"}, 
            {"value": 4.0}
        ]
        result = self.filter_engine.filter(data, "value:[2 TO 4]")
        assert len(result) >= 2  # Should handle numeric comparison


class TestQueryParsingAndAST:
    """Test query parsing and AST manipulation features"""
    
    def setup_method(self):
        self.filter_engine = LuceneObjectFilter()
    
    def test_query_explanation(self):
        """Test query structure explanation"""
        queries = [
            "field:value",
            "field1:value1 AND field2:value2",
            "(field1:value1 OR field2:value2) AND field3:value3",
            "field:[1 TO 10]",
            "field:prefix*"
        ]
        
        for query in queries:
            explanation = self.filter_engine.explain_query(query)
            assert isinstance(explanation, str)
            assert len(explanation) > 0
    
    def test_malformed_query_handling(self):
        """Test various malformed query scenarios"""
        malformed_queries = [
            "field:",           # Missing value
            ":value",           # Missing field
            "field:value AND",  # Incomplete boolean
            "field:[1 TO",      # Incomplete range
            "((field:value",    # Unmatched parentheses
        ]
        
        data = [{"field": "value", "other": "data"}]
        
        for bad_query in malformed_queries:
            # Should not crash
            result = self.filter_engine.filter(data, bad_query)
            assert isinstance(result, list)


# Additional fixtures and utilities for testing
@pytest.fixture
def sample_servers():
    """Pytest fixture providing sample server data"""
    return [
        {"name": "web-01", "cpu": 85, "memory": 16, "status": "running"},
        {"name": "db-01", "cpu": 45, "memory": 32, "status": "running"},
        {"name": "cache-01", "cpu": 25, "memory": 8, "status": "stopped"},
    ]


@pytest.fixture
def lucene_filter():
    """Pytest fixture providing LuceneObjectFilter instance"""
    return LuceneObjectFilter()


@pytest.fixture
def complex_nested_data():
    """Fixture with complex nested data structures"""
    return [
        {
            "server": {
                "name": "web-cluster",
                "nodes": [
                    {"id": "node-1", "cpu": 80, "healthy": True},
                    {"id": "node-2", "cpu": 60, "healthy": False}
                ],
                "config": {
                    "ssl": {"enabled": True, "port": 443},
                    "monitoring": {"enabled": False}
                }
            },
            "tags": {"env": "prod", "team": "platform"}
        }
    ]


def test_fixture_usage(lucene_filter, sample_servers):
    """Example test using fixtures"""
    result = lucene_filter.filter(sample_servers, "status:running")
    assert len(result) == 2


def test_complex_nested_filtering(lucene_filter, complex_nested_data):
    """Test filtering on complex nested structures"""
    result = lucene_filter.filter(complex_nested_data, "server.config.ssl.enabled:true")
    assert len(result) == 1
    
    result = lucene_filter.filter(complex_nested_data, "tags.env:prod")
    assert len(result) == 1


# Performance benchmarking tests
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests (marked for optional execution)"""
    
    def test_large_dataset_benchmark(self):
        """Benchmark filtering on large datasets"""
        import time
        
        filter_engine = LuceneObjectFilter()
        
        # Generate 10k records
        large_dataset = []
        for i in range(10000):
            large_dataset.append({
                "id": i,
                "name": f"item-{i}",
                "category": f"cat-{i % 20}",
                "value": i * 1.5,
                "active": i % 3 == 0
            })
        
        # Benchmark simple query
        start_time = time.time()
        result = filter_engine.filter(large_dataset, "category:cat-5")
        simple_time = time.time() - start_time
        
        # Benchmark complex query
        start_time = time.time()
        result = filter_engine.filter(large_dataset, 
            "active:true AND value:[1000 TO 5000] AND category:cat-*")
        complex_time = time.time() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        assert simple_time < 1.0, f"Simple query took {simple_time:.3f}s"
        assert complex_time < 2.0, f"Complex query took {complex_time:.3f}s"


# Test runner configuration
if __name__ == "__main__":
    # Run tests with pytest
    # Use -m "not performance" to skip performance tests
    pytest.main([__file__, "-v", "--tb=short"])