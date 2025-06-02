import pytest
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

# Import the simple implementation
from simple_lucene_filter import SimpleLuceneFilter, ObjectFilterTransformer


@dataclass
class Server:
    """Test dataclass for server objects"""
    name: str
    cpu: float
    memory: int
    status: str
    tags: List[str]
    created: str
    config: Dict[str, Any]


@dataclass
class Alert:
    """Test dataclass for alert objects"""
    severity: str
    message: str
    resolved: bool
    count: int


class TestSimpleLuceneFilterBasics:
    """Test basic functionality - the 80% use cases"""
    
    def setup_method(self):
        """Setup test data for each test"""
        self.filter = SimpleLuceneFilter()
        
        # Dictionary test data (most common case)
        self.servers = [
            {
                "name": "web-01",
                "cpu": 85.5,
                "memory": 16,
                "status": "running",
                "tags": ["web", "production"],
                "config": {"ssl": True, "port": 443}
            },
            {
                "name": "db-01",
                "cpu": 45.2,
                "memory": 32,
                "status": "running",
                "tags": ["database", "production"],
                "config": {"ssl": False, "port": 5432}
            },
            {
                "name": "cache-01",
                "cpu": 25.0,
                "memory": 8,
                "status": "stopped",
                "tags": ["cache", "development"],
                "config": {"ssl": False, "port": 6379}
            }
        ]
        
        # Dataclass test data
        self.alerts = [
            Alert("critical", "Database connection failed", False, 5),
            Alert("warning", "High CPU usage detected", True, 12),
            Alert("info", "Backup completed successfully", True, 1),
            Alert("critical", "Disk space low", False, 3),
        ]
    
    def test_empty_query_returns_all(self):
        """Empty queries should return all objects"""
        result = self.filter.filter(self.servers, "")
        assert len(result) == len(self.servers)
        
        result = self.filter.filter(self.servers, "   ")
        assert len(result) == len(self.servers)
    
    def test_invalid_query_returns_empty(self):
        """Invalid queries should return empty list (graceful degradation)"""
        result = self.filter.filter(self.servers, "invalid[[query")
        assert result == []
        
        result = self.filter.filter(self.servers, "field:")
        assert result == []
    
    def test_filter_returns_list(self):
        """Filter should always return a list"""
        result = self.filter.filter(self.servers, "cpu:[51 TO *]")
        assert isinstance(result, list)
    
    def test_query_validation(self):
        """Test query validation functionality"""
        # Valid query
        result = self.filter.validate_query("status:running")
        assert result['valid'] is True
        assert result['error'] is None
        assert result['ast'] is not None
        
        # Invalid query
        result = self.filter.validate_query("invalid[[query")
        assert result['valid'] is False
        assert result['error'] is not None
        assert result['ast'] is None


class TestFieldMatching:
    """Test field-based matching - core 80% functionality"""
    
    def setup_method(self):
        self.filter = SimpleLuceneFilter()
        self.data = [
            {"name": "server-1", "cpu": 85, "status": "running", "active": True},
            {"name": "server-2", "cpu": 45, "status": "stopped", "active": False},
            {"name": "server-3", "cpu": 25, "status": "running", "active": True},
        ]
    
    def test_string_field_exact_match(self):
        """Test exact string field matching"""
        result = self.filter.filter(self.data, "status:running")
        assert len(result) == 2
        assert all(obj["status"] == "running" for obj in result)
    
    def test_numeric_field_match(self):
        """Test numeric field matching"""
        result = self.filter.filter(self.data, "cpu:45")
        assert len(result) == 1
        assert result[0]["name"] == "server-2"
        
        # Test float matching
        float_data = [{"value": 3.14}, {"value": 2.71}]
        result = self.filter.filter(float_data, "value:3.14")
        assert len(result) == 1
        assert result[0]["value"] == 3.14
    
    def test_boolean_field_match(self):
        """Test boolean field matching"""
        result = self.filter.filter(self.data, "active:true")
        assert len(result) == 2
        assert all(obj["active"] is True for obj in result)
        
        result = self.filter.filter(self.data, "active:false")
        assert len(result) == 1
        assert result[0]["active"] is False
    
    def test_nonexistent_field(self):
        """Test filtering on nonexistent fields returns empty"""
        result = self.filter.filter(self.data, "nonexistent:value")
        assert len(result) == 0
    
    def test_null_field_handling(self):
        """Test handling of null/None field values"""
        data_with_nulls = [
            {"name": "server1", "value": "present"},
            {"name": "server2", "value": None},
            {"name": "server3"}  # Missing field
        ]
        
        result = self.filter.filter(data_with_nulls, "value:null")
        assert len(result) == 1
        assert result[0]["name"] == "server2"


class TestRangeQueries:
    """Test range queries [min TO max]"""
    
    def setup_method(self):
        self.filter = SimpleLuceneFilter()
        self.data = [
            {"name": "low", "cpu": 10, "memory": 4},
            {"name": "medium", "cpu": 50, "memory": 8},
            {"name": "high", "cpu": 80, "memory": 16},
            {"name": "max", "cpu": 95, "memory": 32},
        ]
    
    def test_numeric_range_inclusive(self):
        """Test inclusive numeric range queries"""
        result = self.filter.filter(self.data, "cpu:[50 TO 80]")
        assert len(result) == 2
        names = [obj["name"] for obj in result]
        assert "medium" in names
        assert "high" in names
    
    def test_numeric_range_exclusive(self):
        """Test exclusive numeric range queries"""
        result = self.filter.filter(self.data, "cpu:{49 TO 81}")
        assert len(result) == 2
        assert all(49 < obj["cpu"] < 81 for obj in result)
    
    def test_open_ended_ranges(self):
        """Test open-ended ranges with *"""
        # Greater than or equal
        result = self.filter.filter(self.data, "cpu:[70 TO *]")
        assert len(result) == 2
        assert all(obj["cpu"] >= 70 for obj in result)
        
        # Less than or equal
        result = self.filter.filter(self.data, "memory:[* TO 10]")
        assert len(result) == 2
        assert all(obj["memory"] <= 10 for obj in result)
    
    def test_string_range_fallback(self):
        """Test string range queries as fallback"""
        string_data = [
            {"name": "apple"}, {"name": "banana"}, {"name": "cherry"}
        ]
        
        result = self.filter.filter(string_data, "name:[b TO c]")
        assert len(result) >= 1  # Should match "banana" and possibly "cherry"


class TestBooleanLogic:
    """Test AND, OR, NOT operations"""
    
    def setup_method(self):
        self.filter = SimpleLuceneFilter()
        self.data = [
            {"name": "web-01", "cpu": 80, "status": "running", "env": "prod"},
            {"name": "web-02", "cpu": 30, "status": "running", "env": "dev"},
            {"name": "db-01", "cpu": 70, "status": "stopped", "env": "prod"},
            {"name": "cache-01", "cpu": 20, "status": "idle", "env": "dev"},
        ]
    
    def test_and_operator(self):
        """Test AND logic"""
        result = self.filter.filter(self.data, "status:running AND env:prod")
        assert len(result) == 1
        assert result[0]["name"] == "web-01"
        
        result = self.filter.filter(self.data, "cpu:[60 TO *] AND status:running")
        assert len(result) == 1
        assert result[0]["name"] == "web-01"
    
    def test_or_operator(self):
        """Test OR logic"""
        result = self.filter.filter(self.data, "status:stopped OR status:idle")
        assert len(result) == 2
        statuses = [obj["status"] for obj in result]
        assert "stopped" in statuses
        assert "idle" in statuses
        
        result = self.filter.filter(self.data, "cpu:[70 TO *] OR cpu:[* TO 25]")
        assert len(result) == 2
    
    def test_not_operator(self):
        """Test NOT logic"""
        result = self.filter.filter(self.data, "NOT status:running")
        assert len(result) == 2
        assert all(obj["status"] != "running" for obj in result)
        
        result = self.filter.filter(self.data, "NOT env:dev")
        assert len(result) == 2
        assert all(obj["env"] != "dev" for obj in result)
    
    def test_complex_boolean_logic(self):
        """Test complex combinations with parentheses"""
        result = self.filter.filter(self.data, 
            "env:prod AND (status:running OR cpu:[65 TO *])")
        assert len(result) == 2  # web-01 and db-01
        
        result = self.filter.filter(self.data,
            "(status:running AND env:prod) OR (status:idle AND env:dev)")
        assert len(result) == 2  # web-01 and cache-01
    
    def test_implicit_and_operation(self):
        """Test implicit AND operations (space between terms)"""
        # luqum's UnknownOperationResolver should handle this
        result = self.filter.filter(self.data, "env:prod status:running")
        assert len(result) == 1
        assert result[0]["name"] == "web-01"


class TestWildcardQueries:
    """Test wildcard pattern matching"""
    
    def setup_method(self):
        self.filter = SimpleLuceneFilter()
        self.data = [
            {"name": "web-server-01", "type": "frontend"},
            {"name": "api-gateway-02", "type": "middleware"},
            {"name": "db-primary", "type": "database"},
            {"name": "cache-redis", "type": "caching"},
        ]
    
    def test_prefix_wildcard(self):
        """Test prefix matching with *"""
        result = self.filter.filter(self.data, "name:web*")
        assert len(result) == 1
        assert result[0]["name"] == "web-server-01"
    
    def test_suffix_wildcard(self):
        """Test suffix matching with *"""
        result = self.filter.filter(self.data, "name:*01")
        assert len(result) == 1
        assert result[0]["name"] == "web-server-01"
    
    def test_contains_wildcard(self):
        """Test contains matching with *"""
        result = self.filter.filter(self.data, "name:*gateway*")
        assert len(result) == 1
        assert result[0]["name"] == "api-gateway-02"
    
    def test_single_char_wildcard(self):
        """Test single character wildcard with ?"""
        result = self.filter.filter(self.data, "name:*-0?")
        assert len(result) == 2  # Matches -01 and -02
    
    def test_multiple_wildcards(self):
        """Test multiple wildcard patterns"""
        result = self.filter.filter(self.data, "name:*-*")
        assert len(result) >= 2  # Should match hyphenated names


class TestNestedFieldAccess:
    """Test nested field access with dot notation"""
    
    def setup_method(self):
        self.filter = SimpleLuceneFilter()
        self.data = [
            {
                "server": {
                    "name": "web-01",
                    "config": {"ssl": True, "port": 443}
                },
                "tags": ["web", "prod"]
            },
            {
                "server": {
                    "name": "db-01", 
                    "config": {"ssl": False, "port": 5432}
                },
                "tags": ["db", "prod"]
            }
        ]
    
    def test_nested_dict_access(self):
        """Test accessing nested dictionary fields"""
        result = self.filter.filter(self.data, "server.name:web-01")
        assert len(result) == 1
        assert result[0]["server"]["name"] == "web-01"
        
        result = self.filter.filter(self.data, "server.config.ssl:true")
        assert len(result) == 1
        assert result[0]["server"]["config"]["ssl"] is True
    
    def test_array_index_access(self):
        """Test accessing array elements by index"""
        result = self.filter.filter(self.data, "tags.0:web")
        assert len(result) == 1
        assert result[0]["tags"][0] == "web"
        
        result = self.filter.filter(self.data, "tags.1:prod")
        assert len(result) == 2  # Both have "prod" at index 1


class TestDataclassObjects:
    """Test filtering dataclass objects"""
    
    def setup_method(self):
        self.filter = SimpleLuceneFilter()
        self.servers = [
            Server("web-01", 85.5, 16, "running", ["web", "prod"], 
                   "2024-01-15", {"ssl": True, "backup": False}),
            Server("db-01", 45.2, 32, "running", ["db", "prod"], 
                   "2024-01-10", {"ssl": False, "backup": True}),
            Server("cache-01", 25.0, 8, "stopped", ["cache", "dev"], 
                   "2024-01-20", {"ssl": False, "backup": False}),
        ]
    
    def test_dataclass_field_access(self):
        """Test accessing dataclass fields"""
        result = self.filter.filter(self.servers, "status:running")
        assert len(result) == 2
        assert all(server.status == "running" for server in result)
        
        result = self.filter.filter(self.servers, "cpu:85.5")
        assert len(result) == 1
        assert result[0].name == "web-01"
    
    def test_dataclass_nested_access(self):
        """Test accessing nested fields in dataclass"""
        result = self.filter.filter(self.servers, "config.ssl:true")
        assert len(result) == 1
        assert result[0].name == "web-01"
        
        result = self.filter.filter(self.servers, "tags.0:web")
        assert len(result) == 1
        assert result[0].name == "web-01"


class TestFullTextSearch:
    """Test full-text search functionality"""
    
    def setup_method(self):
        self.filter = SimpleLuceneFilter()
        self.data = [
            {"name": "web-server", "description": "Handles HTTP requests"},
            {"name": "database", "description": "Stores application data"},
            {"name": "cache", "description": "Redis cache for performance"},
        ]
    
    def test_fulltext_search(self):
        """Test searching across all fields"""
        result = self.filter.filter(self.data, "HTTP")
        assert len(result) == 1
        assert result[0]["name"] == "web-server"
        
        result = self.filter.filter(self.data, "data")
        assert len(result) >= 1  # Should match "database" and possibly "application data"
    
    def test_case_insensitive_search(self):
        """Test case-insensitive search"""
        result = self.filter.filter(self.data, "redis")
        assert len(result) == 1
        assert result[0]["name"] == "cache"
    
    def test_phrase_search(self):
        """Test quoted phrase search"""
        result = self.filter.filter(self.data, '"HTTP requests"')
        assert len(result) == 1
        assert result[0]["name"] == "web-server"


class TestFuzzySearch:
    """Test fuzzy search functionality"""
    
    def setup_method(self):
        self.filter = SimpleLuceneFilter()
        self.data = [
            {"name": "test-server", "description": "testing server"},
            {"name": "prod-server", "description": "production server"},
            {"name": "dev-machine", "description": "development machine"},
        ]
    
    def test_fuzzy_field_search(self):
        """Test fuzzy matching on specific fields"""
        result = self.filter.filter(self.data, "name:test~")
        assert len(result) >= 1  # Should match "test-server"
    
    def test_fuzzy_fulltext_search(self):
        """Test fuzzy full-text search"""
        result = self.filter.filter(self.data, "produc~")
        assert len(result) >= 1  # Should match "production"


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        self.filter = SimpleLuceneFilter()
    
    def test_empty_data_list(self):
        """Test filtering empty data list"""
        result = self.filter.filter([], "field:value")
        assert result == []
    
    def test_none_objects_in_list(self):
        """Test handling None objects in list"""
        data = [
            {"name": "valid"},
            None,
            {"name": "also-valid"}
        ]
        
        # Filter should skip None objects gracefully
        result = self.filter.filter(data, "name:valid")
        assert len(result) == 1
        assert result[0]["name"] == "valid"
    
    def test_objects_with_none_values(self):
        """Test objects with None field values"""
        data = [
            {"name": "server1", "value": None},
            {"name": "server2", "value": "present"},
        ]
        
        result = self.filter.filter(data, "value:null")
        assert len(result) == 1
        assert result[0]["name"] == "server1"
    
    def test_mixed_object_types(self):
        """Test filtering mixed object types"""
        from collections import namedtuple
        
        SimpleObj = namedtuple('SimpleObj', ['name', 'value'])
        
        mixed_data = [
            {"name": "dict", "value": 1},
            SimpleObj("tuple", 2),
            Server("dataclass", 3.0, 16, "running", [], "2024-01-01", {})
        ]
        
        # Should handle different object types gracefully
        result = self.filter.filter(mixed_data, "name:dict")
        assert len(result) == 1
        
        result = self.filter.filter(mixed_data, "name:tuple")
        assert len(result) == 1
        
        result = self.filter.filter(mixed_data, "name:dataclass")
        assert len(result) == 1
    
    def test_deeply_nested_objects(self):
        """Test handling deeply nested structures"""
        deep_data = [
            {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "value": "deep"
                            }
                        }
                    }
                }
            }
        ]
        
        # Should handle reasonable nesting
        result = self.filter.filter(deep_data, "level1.level2.level3.level4.value:deep")
        assert len(result) == 1
    
    def test_large_objects(self):
        """Test handling objects with many fields"""
        large_obj = {f"field_{i}": f"value_{i}" for i in range(100)}
        large_obj["target"] = "findme"
        
        data = [large_obj]
        
        result = self.filter.filter(data, "target:findme")
        assert len(result) == 1
    
    def test_circular_reference_prevention(self):
        """Test that circular references don't cause infinite loops"""
        obj1 = {"name": "obj1", "ref": None}
        obj2 = {"name": "obj2", "ref": obj1}
        obj1["ref"] = obj2  # Create circular reference
        
        data = [obj1, obj2]
        
        # Should handle without infinite recursion
        result = self.filter.filter(data, "name:obj1")
        assert len(result) == 1


class TestQueryValidation:
    """Test query validation functionality"""
    
    def setup_method(self):
        self.filter = SimpleLuceneFilter()
    
    def test_valid_queries(self):
        """Test validation of valid queries"""
        valid_queries = [
            "field:value",
            "field1:value1 AND field2:value2", 
            "field:[1 TO 10]",
            "field:prefix*",
            "(field1:value1 OR field2:value2) AND field3:value3"
        ]
        
        for query in valid_queries:
            result = self.filter.validate_query(query)
            assert result['valid'] is True, f"Query should be valid: {query}"
            assert result['error'] is None
            assert result['ast'] is not None
    
    def test_invalid_queries(self):
        """Test validation of invalid queries"""
        invalid_queries = [
            "field:",
            ":value", 
            "field:value AND",
            "field:[1 TO",
            "((field:value"
        ]
        
        for query in invalid_queries:
            result = self.filter.validate_query(query)
            assert result['valid'] is False, f"Query should be invalid: {query}"
            assert result['error'] is not None
            assert result['ast'] is None


# Integration tests for real-world scenarios
class TestRealWorldScenarios:
    """Test realistic usage patterns"""
    
    def setup_method(self):
        self.filter = SimpleLuceneFilter()
        
        # Realistic server monitoring data
        self.servers = [
            {
                "hostname": "web-prod-01.company.com",
                "ip": "10.0.1.10",
                "cpu_percent": 78.5,
                "memory_gb": 16,
                "disk_usage": {"root": 45, "var": 67},
                "services": [
                    {"name": "nginx", "status": "running", "port": 80},
                    {"name": "app", "status": "running", "port": 8080}
                ],
                "tags": ["web", "production", "zone-a"],
                "last_check": "2024-01-15T10:30:00Z",
                "alerts": ["high_cpu", "disk_warning"]
            },
            {
                "hostname": "db-prod-01.company.com", 
                "ip": "10.0.2.10",
                "cpu_percent": 45.2,
                "memory_gb": 64,
                "disk_usage": {"root": 23, "data": 89},
                "services": [
                    {"name": "postgresql", "status": "running", "port": 5432},
                    {"name": "pgbouncer", "status": "running", "port": 6432}
                ],
                "tags": ["database", "production", "zone-b"],
                "last_check": "2024-01-15T10:28:00Z", 
                "alerts": ["disk_critical"]
            },
            {
                "hostname": "cache-dev-01.company.com",
                "ip": "10.0.3.10", 
                "cpu_percent": 25.0,
                "memory_gb": 8,
                "disk_usage": {"root": 15},
                "services": [
                    {"name": "redis", "status": "running", "port": 6379}
                ],
                "tags": ["cache", "development", "zone-a"],
                "last_check": "2024-01-15T10:25:00Z",
                "alerts": []
            }
        ]
    
    def test_server_monitoring_queries(self):
        """Test common server monitoring queries"""
        # Find high CPU servers
        result = self.filter.filter(self.servers, "cpu_percent:[70 TO *]")
        assert len(result) == 1
        assert result[0]["hostname"] == "web-prod-01.company.com"
        
        # Find production servers with alerts
        result = self.filter.filter(self.servers, "tags:production AND alerts:*")
        assert len(result) == 2
        
        # Find servers in specific zone with specific service
        result = self.filter.filter(self.servers, "tags:zone-a AND services.0.name:nginx")
        assert len(result) == 1
        
        # Complex operational query
        result = self.filter.filter(self.servers, 
            "(cpu_percent:[60 TO *] OR disk_usage.data:[80 TO *]) AND tags:production")
        assert len(result) == 2  # High CPU web server and high disk DB server
    
    def test_log_analysis_scenario(self):
        """Test log entry filtering scenario"""
        log_entries = [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "level": "ERROR",
                "service": "api-gateway",
                "message": "Database connection timeout",
                "request_id": "req-123",
                "user_id": "user-456",
                "response_time_ms": 5000
            },
            {
                "timestamp": "2024-01-15T10:31:00Z", 
                "level": "WARN",
                "service": "user-service",
                "message": "High memory usage detected",
                "request_id": "req-124", 
                "user_id": "user-789",
                "response_time_ms": 1200
            },
            {
                "timestamp": "2024-01-15T10:32:00Z",
                "level": "INFO", 
                "service": "api-gateway",
                "message": "Request processed successfully",
                "request_id": "req-125",
                "user_id": "user-456", 
                "response_time_ms": 150
            }
        ]
        
        # Find error logs
        result = self.filter.filter(log_entries, "level:ERROR")
        assert len(result) == 1
        
        # Find slow requests
        result = self.filter.filter(log_entries, "response_time_ms:[1000 TO *]")
        assert len(result) == 2
        
        # Find logs for specific user
        result = self.filter.filter(log_entries, "user_id:user-456")
        assert len(result) == 2
        
        # Complex log query
        result = self.filter.filter(log_entries, 
            "service:api-gateway AND (level:ERROR OR response_time_ms:[3000 TO *])")
        assert len(result) == 1


# Performance tests (simple ones for the 80/20 rule)
class TestSimplePerformance:
    """Basic performance tests for typical use cases"""
    
    def test_medium_dataset_performance(self):
        """Test with realistic dataset size (few hundred objects)"""
        import time
        
        filter_obj = SimpleLuceneFilter()
        
        # Generate 500 objects
        large_dataset = []
        for i in range(500):
            large_dataset.append({
                "id": i,
                "name": f"server-{i:03d}",
                "cpu": i % 100,
                "status": "running" if i % 2 == 0 else "stopped",
                "tags": ["web" if i % 3 == 0 else "db", "prod" if i % 4 == 0 else "dev"]
            })
        
        # Test simple query
        start_time = time.time()
        result = filter_obj.filter(large_dataset, "status:running")
        simple_time = time.time() - start_time
        
        assert len(result) == 250  # Half should be running
        assert simple_time < 1.0  # Should be fast for 500 objects
        
        # Test complex query
        start_time = time.time()
        result = filter_obj.filter(large_dataset, 
            "status:running AND (cpu:[70 TO *] OR tags:web)")
        complex_time = time.time() - start_time
        
        assert complex_time < 2.0  # Should still be reasonable
        assert len(result) > 0


# Test runner configuration
if __name__ == "__main__":
    # Run tests with simple configuration
    pytest.main([
        __file__, 
        "-v",
        "--tb=short"
    ])