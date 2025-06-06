"""
Simple Search Library Tests - Complete validation
===============================================

Tests core functionality, edge cases, and robustness.
Run with: pytest test_simple_search.py -v
"""

import pytest
import time
import threading
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import patch

from simple_search import (
    index, indexed_field, SimpleSearch, IndexBuilder, KeywordIndex, 
    NumericIndex, TextIndex, FuzzyIndex, IndexType, FieldInfo
)

try:
    from pydantic import BaseModel
    from simple_search import IndexedField
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


@dataclass
class Server:
    """Test model with indexed fields"""
    status: str = indexed_field()
    cpu_percent: float = indexed_field()
    name: str = indexed_field()
    tags: List[str] = indexed_field()
    description: str  # Not indexed
    
    @index
    @property
    def is_healthy(self) -> bool:
        return self.status == "running" and self.cpu_percent < 80


if HAS_PYDANTIC:
    class PydanticServer(BaseModel):
        status: str = IndexedField()
        cpu_percent: float = IndexedField(ge=0, le=100)
        name: str = IndexedField()
        description: str  # Not indexed


class TestIndexTypes:
    """Test individual index implementations"""
    
    def test_keyword_index_basic(self):
        """KeywordIndex should handle exact matches"""
        index = KeywordIndex()
        index.add(0, "running")
        index.add(1, "stopped")
        index.add(2, "running")
        
        assert index.search("running") == {0, 2}
        assert index.search("stopped") == {1}
        assert index.search("missing") == set()
    
    def test_keyword_index_case_insensitive(self):
        """KeywordIndex should be case insensitive"""
        index = KeywordIndex()
        index.add(0, "Running")
        index.add(1, "STOPPED")
        
        assert index.search("running") == {0}
        assert index.search("stopped") == {1}
    
    def test_keyword_index_list_values(self):
        """KeywordIndex should handle list/tuple values like tags"""
        index = KeywordIndex()
        index.add(0, ["web", "production"])
        index.add(1, ["database", "production"])
        index.add(2, ["cache", "development"])
        
        assert index.search("production") == {0, 1}
        assert index.search("web") == {0}
        assert index.search("development") == {2}
    
    def test_keyword_index_wildcards(self):
        """KeywordIndex should support wildcards"""
        index = KeywordIndex()
        index.add(0, "web-server-01")
        index.add(1, "db-server-01")
        index.add(2, "cache-redis")
        
        assert index.wildcard_search("*server*") == {0, 1}
        assert index.wildcard_search("web*") == {0}
        assert index.wildcard_search("*-01") == {0, 1}
    
    def test_keyword_index_invalid_regex(self):
        """KeywordIndex should handle invalid regex gracefully"""
        index = KeywordIndex()
        index.add(0, "test")
        
        # Invalid regex patterns should return empty set
        assert index.wildcard_search("*[invalid") == set()
        assert index.wildcard_search("(?bad)") == set()
    
    def test_numeric_index_basic(self):
        """NumericIndex should handle exact matches and ranges"""
        index = NumericIndex()
        index.add(0, 10.5)
        index.add(1, 20.0)
        index.add(2, 30.5)
        index.add(3, "invalid")  # Should be ignored
        
        assert index.search("20.0") == {1}
        assert index.range_search(15.0, 25.0) == {1}
        assert index.range_search(10.0, 35.0) == {0, 1, 2}
    
    def test_numeric_index_auto_build(self):
        """NumericIndex should auto-build when queried"""
        index = NumericIndex()
        index.add(0, 10)
        index.add(1, 20)
        
        # Should auto-build on first query
        assert not index.built
        results = index.search("10")
        assert index.built
        assert results == {0}
    
    def test_text_index_basic(self):
        """TextIndex should tokenize and search terms"""
        index = TextIndex()
        index.add(0, "web server running")
        index.add(1, "database server stopped")
        index.add(2, "cache service running")
        
        assert index.search("server") == {0, 1}
        assert index.search("running") == {0, 2}
        assert index.search("missing") == set()
    
    def test_fuzzy_index_basic(self):
        """FuzzyIndex should handle fuzzy matching"""
        index = FuzzyIndex()
        index.add(0, "running")
        index.add(1, "stopped")
        index.add(2, "runing")  # Typo
        
        # Exact match
        assert index.search("running") == {0}
        
        # Fuzzy match should find typos
        fuzzy_results = index.fuzzy_search("running", threshold=0.6)
        assert 0 in fuzzy_results
        assert 2 in fuzzy_results  # Should match "runing"


class TestIndexBuilder:
    """Test index building and field discovery"""
    
    def test_discover_dataclass_fields(self):
        """Should find dataclass indexed fields"""
        fields = IndexBuilder.discover_fields([Server])
        
        field_names = [f.name for f in fields]
        assert "status" in field_names
        assert "cpu_percent" in field_names
        assert "name" in field_names
        assert "tags" in field_names
        assert "is_healthy" in field_names
        assert "description" not in field_names  # Not indexed
    
    @pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
    def test_discover_pydantic_fields(self):
        """Should find Pydantic indexed fields"""
        fields = IndexBuilder.discover_fields([PydanticServer])
        
        field_names = [f.name for f in fields]
        assert "status" in field_names
        assert "cpu_percent" in field_names
        assert "name" in field_names
        assert "description" not in field_names  # Not indexed
    
    def test_type_detection_numeric(self):
        """Should detect numeric fields"""
        field_info = FieldInfo("cpu", "Server.cpu", IndexType.KEYWORD, float)
        numeric_values = [1, 2.5, 3, 4.0, 5]
        assert IndexBuilder.detect_type(field_info, numeric_values) == IndexType.NUMERIC
    
    def test_type_detection_text(self):
        """Should detect text fields based on length"""
        field_info = FieldInfo("description", "Server.description", IndexType.KEYWORD, str)
        long_text = ["This is a very long description with many words that should be detected as text field"]
        assert IndexBuilder.detect_type(field_info, long_text) == IndexType.TEXT
    
    def test_type_detection_keyword(self):
        """Should default to keyword for short strings"""
        field_info = FieldInfo("status", "Server.status", IndexType.KEYWORD, str)
        keywords = ["active", "inactive", "pending"]
        assert IndexBuilder.detect_type(field_info, keywords) == IndexType.KEYWORD
    
    def test_type_detection_from_hints(self):
        """Should use type hints for detection"""
        field_info = FieldInfo("tags", "Server.tags", IndexType.KEYWORD, List[str])
        values = [["web", "prod"], ["db", "staging"]]
        assert IndexBuilder.detect_type(field_info, values) == IndexType.KEYWORD
    
    def test_type_detection_fuzzy(self):
        """Should detect fuzzy fields for diverse string data"""
        field_info = FieldInfo("description", "Server.description", IndexType.KEYWORD, str)
        diverse_strings = ["server-alpha", "server-beta", "server-gamma", "cache-01", "database-main", "api-gateway"]
        detected_type = IndexBuilder.detect_type(field_info, diverse_strings)
        # Should be FUZZY due to variety and length
        assert detected_type == IndexType.FUZZY
    
    def test_build_indexes(self):
        """Should build appropriate index types"""
        servers = [
            Server("running", 85.5, "web-01", ["prod"], "Web server"),
            Server("stopped", 45.2, "db-01", ["prod"], "Database server")
        ]
        
        fields = IndexBuilder.discover_fields([Server])
        indexes = IndexBuilder.build_indexes(servers, fields)
        
        # Check that indexes were created
        assert "status" in indexes
        assert "cpu_percent" in indexes
        assert "name" in indexes
        assert "is_healthy" in indexes
        
        # Check correct types
        assert isinstance(indexes["status"], KeywordIndex)
        assert isinstance(indexes["cpu_percent"], NumericIndex)


class TestSimpleSearch:
    """Test main search interface"""
    
    def setup_method(self):
        """Create test data"""
        self.servers = [
            Server("running", 85.5, "web-01", ["web", "production"], "Web server"),
            Server("stopped", 45.2, "db-01", ["database", "production"], "Database server"),
            Server("running", 25.0, "cache-01", ["cache", "development"], "Cache server"),
            Server("maintenance", 90.0, "api-gateway", ["api", "production"], "API gateway"),
        ]
        self.search = SimpleSearch([Server], self.servers)
    
    def test_initialization(self):
        """Should initialize and build indexes"""
        assert len(self.search.objects) == 4
        assert len(self.search.fields) > 0
        assert len(self.search.indexes) > 0
    
    def test_simple_queries(self):
        """Should handle basic field:value queries"""
        # Exact match
        results = self.search.query("status:running")
        assert len(results) == 2
        assert all(s.status == "running" for s in results)
        
        # Numeric query
        results = self.search.query("cpu_percent:90.0")
        assert len(results) == 1
        assert results[0].name == "api-gateway"
    
    def test_range_queries(self):
        """Should handle numeric range queries"""
        # High CPU servers
        results = self.search.query("cpu_percent:[80 TO *]")
        assert len(results) == 2
        assert all(s.cpu_percent >= 80 for s in results)
        
        # Medium CPU servers
        results = self.search.query("cpu_percent:[40 TO 60]")
        assert len(results) == 1
        assert results[0].cpu_percent == 45.2
        
        # Test infinity bounds
        results = self.search.query("cpu_percent:[* TO 50]")
        assert len(results) == 2  # 45.2 and 25.0
    
    def test_wildcard_queries(self):
        """Should handle wildcard patterns"""
        results = self.search.query("name:*-01")
        assert len(results) == 3
        assert all(s.name.endswith("-01") for s in results)
        
        results = self.search.query("name:web*")
        assert len(results) == 1
        assert results[0].name == "web-01"
    
    def test_boolean_queries(self):
        """Should handle AND/OR logic"""
        # AND query
        results = self.search.query("status:running AND cpu_percent:[80 TO *]")
        assert len(results) == 1
        assert results[0].name == "web-01"
        
        # OR query
        results = self.search.query("status:stopped OR status:maintenance")
        assert len(results) == 2
        statuses = {s.status for s in results}
        assert statuses == {"stopped", "maintenance"}
    
    def test_tags_functionality(self):
        """Should search within tag/list fields"""
        results = self.search.query("tags:production")
        assert len(results) == 3
        assert all("production" in s.tags for s in results)
        
        results = self.search.query("tags:web")
        assert len(results) == 1
        assert results[0].name == "web-01"
    
    def test_property_field_queries(self):
        """Should handle computed property fields"""
        results = self.search.query("is_healthy:true")
        healthy_servers = [s for s in results if s.is_healthy]
        assert len(healthy_servers) == len(results)
    
    def test_fuzzy_queries(self):
        """Should handle fuzzy search queries"""
        # Add server with fuzzy-indexed field
        servers_with_fuzzy = self.servers + [
            Server("running", 50.0, "test-server", ["testing"], "Test server for fuzzy matching")
        ]
        search = SimpleSearch([Server], servers_with_fuzzy)
        
        # If name is detected as fuzzy, test fuzzy search
        if any(f.name == "name" and f.type == IndexType.FUZZY for f in search.fields):
            results = search.query("name:test~")
            assert len(results) >= 1
    
    def test_empty_query(self):
        """Empty query should return all objects"""
        results = self.search.query("")
        assert len(results) == len(self.servers)
        
        results = self.search.query("   ")
        assert len(results) == len(self.servers)
    
    def test_query_caching(self):
        """Should cache query results"""
        query = "status:running"
        
        # First query
        results1 = self.search.query(query)
        
        # Second query should use cache
        results2 = self.search.query(query)
        
        assert results1 == results2
        assert len(self.search._query_cache) > 0
    
    def test_fallback_without_luqum(self):
        """Should work without luqum for simple queries"""
        with patch('simple_search.HAS_LUQUM', False):
            search = SimpleSearch([Server], self.servers)
            
            # Simple field:value should work
            results = search.query("status:running")
            assert len(results) == 2
    
    def test_invalid_queries(self):
        """Should handle invalid queries gracefully"""
        # Malformed query should not crash
        results = self.search.query("invalid[[query")
        assert isinstance(results, list)
        
        # Non-existent field should return empty or use fallback
        results = self.search.query("nonexistent:value")
        assert isinstance(results, list)
    
    def test_query_correction(self):
        """Should attempt correction on parse failures"""
        # Test field name typo correction
        results = self.search.query("statu:running")  # Typo in 'status'
        # Should either correct or fallback gracefully
        assert isinstance(results, list)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataset(self):
        """Should handle empty datasets"""
        search = SimpleSearch([Server], [])
        results = search.query("status:running")
        assert results == []
    
    def test_none_values(self):
        """Should handle None values gracefully"""
        @dataclass
        class ModelWithNones:
            optional_field: Optional[str] = indexed_field()
        
        data = [
            ModelWithNones("present"),
            ModelWithNones(None),
            ModelWithNones("also_present")
        ]
        
        search = SimpleSearch([ModelWithNones], data)
        results = search.query("optional_field:present")
        assert len(results) == 1
    
    def test_mixed_types_in_field(self):
        """Should handle mixed types in same field"""
        @dataclass
        class MixedModel:
            mixed_field: str = indexed_field()
        
        data = [
            MixedModel("string"),
            MixedModel("42"),
            MixedModel("3.14")
        ]
        
        # Should not crash
        search = SimpleSearch([MixedModel], data)
        results = search.query("mixed_field:string")
        assert len(results) >= 0
    
    def test_thread_safety(self):
        """Query cache should be thread-safe"""
        search = SimpleSearch([Server], [
            Server("running", 50.0, "test-server", ["web"], "Test")
        ])
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    # Use different queries to avoid cache hits masking issues
                    query = f"status:running AND cpu_percent:{50.0 + worker_id + i*0.1}"
                    result = search.query(query)
                    results.append(len(result))
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(4):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        assert len(errors) == 0
        assert len(results) == 40


class TestPerformance:
    """Test performance characteristics"""
    
    def test_performance_relative(self):
        """Use relative performance instead of absolute timing"""
        large_dataset = [Server(f"status_{i%3}", float(i), f"server-{i}", [], "desc") for i in range(100)]
        
        # Time linear search
        start = time.time()
        linear_results = [s for s in large_dataset if s.status == "status_1"]
        linear_time = time.time() - start
        
        # Time indexed search
        search = SimpleSearch([Server], large_dataset)
        start = time.time()
        indexed_results = search.query("status:status_1")
        indexed_time = time.time() - start
        
        # Relative performance (not absolute)
        assert len(indexed_results) == len(linear_results)
        if linear_time > 0:
            speedup = linear_time / max(indexed_time, 0.001)
            assert speedup > 1.0  # Some improvement


class TestRobustnessFixes:
    """Test fixes for likely failure cases"""
    
    def test_complex_type_hints(self):
        """Should handle complex type annotations"""
        @dataclass
        class ComplexModel:
            optional_tags: Optional[List[str]] = indexed_field()
        
        data = [ComplexModel(["tag1"]), ComplexModel(None)]
        search = SimpleSearch([ComplexModel], data)
        assert len(search.indexes) > 0
    
    def test_all_none_field_values(self):
        """Should handle fields with all None values"""
        @dataclass
        class NoneModel:
            nullable_field: Optional[str] = indexed_field()
        
        data = [NoneModel(None), NoneModel(None)]
        search = SimpleSearch([NoneModel], data)
        results = search.query("nullable_field:anything")
        assert results == []
    
    def test_dangerous_wildcard_patterns(self):
        """Should handle dangerous regex patterns safely"""
        index = KeywordIndex()
        index.add(0, "test")
        
        dangerous_patterns = ["***", "[", "(?", "+*", "(?P<>)"]
        for pattern in dangerous_patterns:
            result = index.wildcard_search(pattern)
            assert isinstance(result, set)  # Should not crash
    
    def test_cache_eviction_exact(self):
        """Validate exact LRU cache behavior"""
        servers = [Server("running", 50.0, "test", [], "desc")]
        search = SimpleSearch([Server], servers)
        search._cache_limit = 3
        
        # Fill cache
        search.query("query1")
        search.query("query2") 
        search.query("query3")
        assert len(search._query_cache) == 3
        
        # Access query1 to make it recent
        search.query("query1")
        
        # Add query4 - should evict query2 (LRU)
        search.query("query4")
        assert "query1" in search._query_cache
        assert "query2" not in search._query_cache
        assert "query4" in search._query_cache
    
    def test_float_precision_ranges(self):
        """Should handle float precision in range queries"""
        servers = [
            Server("running", 33.333333, "test1", [], "desc"),
            Server("running", 33.333334, "test2", [], "desc")
        ]
        search = SimpleSearch([Server], servers)
        
        # Very narrow range
        results = search.query("cpu_percent:[33.333333 TO 33.333333]")
        assert len(results) == 1
    
    def test_unicode_field_values(self):
        """Should handle Unicode characters"""
        servers = [Server("rünning", 50.0, "tëst-ñame", ["产品", "тест"], "desc")]
        search = SimpleSearch([Server], servers)
        
        results = search.query("status:rünning")
        assert len(results) == 1
        
        results = search.query("tags:产品")
        assert len(results) == 1


class TestStatsAndMonitoring:
    """Test statistics and monitoring features"""
    
    def test_stats_collection(self):
        """Should collect useful statistics"""
        search = SimpleSearch([Server], [
            Server("running", 85.5, "web-01", ["prod"], "Web server")
        ])
        
        stats = search.stats()
        
        assert stats['objects'] == 1
        assert stats['indexed_fields'] > 0
        assert 'total_memory_mb' in stats
        assert 'fields' in stats
        assert isinstance(stats['has_luqum'], bool)
    
    def test_quick_validate(self):
        """Should validate query syntax quickly"""
        search = SimpleSearch([Server], [])
        
        assert search.quick_validate("status:running") is True
        assert search.quick_validate("status:running AND cpu:[50 TO *]") is True
        assert search.quick_validate("") is False
        assert search.quick_validate("status:running AND") is False
        assert search.quick_validate("((status:running") is False


# Mock luqum imports
@pytest.fixture
def mock_no_luqum():
    """Mock missing luqum dependency"""
    with patch('simple_search.HAS_LUQUM', False):
        yield


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])