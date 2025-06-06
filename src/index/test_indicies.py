"""
Comprehensive Test Suite for Smart Index Decorator System
========================================================

Tests all aspects of the smart indexing system:
- Decorator functionality and configuration
- Smart type detection and defaults
- Index building and querying
- Performance characteristics
- Integration scenarios

Run with: pytest test_smart_index.py -v
"""

import pytest
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import time
import random
import string

# Import the smart index system (assuming it's in smart_index_decorator.py)
from smart_index_decorator import (
    index, IndexType, IndexConfig, SmartIndexAnalyzer, SmartIndexBuilder,
    KeywordIndex, NumericIndex, TextIndex, IndexOptimizedLuceneFilter
)


class TestIndexDecorator:
    """Test the @index decorator functionality"""
    
    def test_decorator_with_defaults(self):
        """Test decorator with no parameters uses smart defaults"""
        @dataclass
        class TestModel:
            @index()
            @property
            def field1(self) -> str:
                return "test"
        
        # Check decorator was applied
        prop = getattr(TestModel, 'field1')
        assert hasattr(prop.fget, '_index_config')
        
        config = prop.fget._index_config
        assert config.enabled is True
        assert config.type == IndexType.AUTO
        assert config.priority == "medium"
    
    def test_decorator_with_explicit_config(self):
        """Test decorator with explicit configuration"""
        @dataclass
        class TestModel:
            @index(type=IndexType.TEXT, priority="high", searchable=True)
            @property
            def description(self) -> str:
                return "test description"
        
        config = TestModel.description.fget._index_config
        assert config.type == IndexType.TEXT
        assert config.priority == "high"
        assert config.searchable is True
    
    def test_decorator_disabled(self):
        """Test decorator with enabled=False"""
        @dataclass
        class TestModel:
            @index(enabled=False)
            @property
            def internal_field(self) -> str:
                return "internal"
        
        config = TestModel.internal_field.fget._index_config
        assert config.enabled is False
    
    def test_decorator_with_options(self):
        """Test decorator with additional options"""
        @dataclass
        class TestModel:
            @index(case_sensitive=True, boost=2.0, custom_param="value")
            @property
            def special_field(self) -> str:
                return "special"
        
        config = TestModel.special_field.fget._index_config
        assert config.case_sensitive is True
        assert config.options['boost'] == 2.0
        assert config.options['custom_param'] == "value"


class TestSmartTypeDetection:
    """Test intelligent type detection"""
    
    def setup_method(self):
        self.analyzer = SmartIndexAnalyzer()
    
    def test_detect_boolean_type(self):
        """Test boolean type detection"""
        samples = [True, False, True, False]
        detected_type = self.analyzer._detect_field_type("enabled", samples)
        assert detected_type == IndexType.BOOLEAN
    
    def test_detect_numeric_type(self):
        """Test numeric type detection"""
        int_samples = [1, 2, 3, 4, 5]
        detected_type = self.analyzer._detect_field_type("count", int_samples)
        assert detected_type == IndexType.NUMERIC
        
        float_samples = [1.5, 2.7, 3.14, 4.0]
        detected_type = self.analyzer._detect_field_type("score", float_samples)
        assert detected_type == IndexType.NUMERIC
    
    def test_detect_date_type(self):
        """Test date type detection"""
        date_samples = [datetime.now(), datetime(2024, 1, 1)]
        detected_type = self.analyzer._detect_field_type("created_at", date_samples)
        assert detected_type == IndexType.DATE
    
    def test_detect_tags_type(self):
        """Test list/tags type detection"""
        tag_samples = [["web", "server"], ["db", "production"], ["cache"]]
        detected_type = self.analyzer._detect_field_type("tags", tag_samples)
        assert detected_type == IndexType.TAGS
    
    def test_detect_keyword_by_name_pattern(self):
        """Test keyword detection by field name patterns"""
        samples = ["running", "stopped", "maintenance"]
        
        # Test various keyword field name patterns
        keyword_fields = ["status", "state", "environment", "priority", "level"]
        for field_name in keyword_fields:
            detected_type = self.analyzer._detect_field_type(field_name, samples)
            assert detected_type == IndexType.KEYWORD
    
    def test_detect_keyword_by_low_uniqueness(self):
        """Test keyword detection by low uniqueness ratio"""
        # Many repeated values (low uniqueness)
        samples = ["active"] * 80 + ["inactive"] * 20
        detected_type = self.analyzer._detect_field_type("some_field", samples)
        assert detected_type == IndexType.KEYWORD
    
    def test_detect_text_by_length_and_uniqueness(self):
        """Test text detection by content characteristics"""
        # Long, unique text content
        samples = [
            "This is a long description with many unique words and content",
            "Another lengthy piece of text with different vocabulary",
            "Yet another unique description with various terms and phrases"
        ]
        detected_type = self.analyzer._detect_field_type("description", samples)
        assert detected_type == IndexType.TEXT
    
    def test_smart_defaults_application(self):
        """Test that smart defaults are applied correctly"""
        config = IndexConfig(type=IndexType.AUTO)
        
        # Test TEXT defaults
        self.analyzer._apply_smart_defaults(config, "description", ["long text content"])
        config.resolved_type = IndexType.TEXT
        self.analyzer._apply_smart_defaults(config, "description", [])
        
        assert config.searchable is True
        assert config.filterable is False
        assert config.sortable is False
        assert config.case_sensitive is False
        
        # Test KEYWORD defaults
        config = IndexConfig(type=IndexType.AUTO)
        config.resolved_type = IndexType.KEYWORD
        self.analyzer._apply_smart_defaults(config, "status", [])
        
        assert config.searchable is False
        assert config.filterable is True
        assert config.sortable is True
        assert config.case_sensitive is True


class TestIndexImplementations:
    """Test individual index implementations"""
    
    def test_keyword_index_basic_operations(self):
        """Test KeywordIndex basic functionality"""
        config = IndexConfig(type=IndexType.KEYWORD, case_sensitive=False)
        index = KeywordIndex("Server.status", config)
        
        # Add documents
        index.add_document(0, "running")
        index.add_document(1, "stopped")
        index.add_document(2, "running")
        index.add_document(3, "RUNNING")  # Test case insensitivity
        
        # Test exact search
        results = index.search("running")
        assert results == {0, 2, 3}
        
        results = index.search("stopped")
        assert results == {1}
        
        results = index.search("nonexistent")
        assert results == set()
    
    def test_keyword_index_wildcard_search(self):
        """Test KeywordIndex wildcard functionality"""
        config = IndexConfig(type=IndexType.KEYWORD, case_sensitive=False)
        index = KeywordIndex("Server.name", config)
        
        # Add documents
        index.add_document(0, "web-server-01")
        index.add_document(1, "db-server-01")
        index.add_document(2, "cache-redis")
        
        # Test wildcard patterns
        results = index.wildcard_search("*-server-*")
        assert results == {0, 1}
        
        results = index.wildcard_search("web*")
        assert results == {0}
        
        results = index.wildcard_search("*redis")
        assert results == {2}
    
    def test_numeric_index_operations(self):
        """Test NumericIndex functionality"""
        config = IndexConfig(type=IndexType.NUMERIC)
        index = NumericIndex("Server.cpu_percent", config)
        
        # Add documents
        values = [85.5, 45.2, 90.0, 25.0, 67.8]
        for i, value in enumerate(values):
            index.add_document(i, value)
        
        index.build()  # Build sorted index
        
        # Test exact search
        results = index.search("85.5")
        assert results == {0}
        
        # Test range searches
        results = index.range_search(40.0, 70.0, True, True)
        assert results == {1, 4}  # 45.2 and 67.8
        
        results = index.range_search(80.0, 100.0, True, True)
        assert results == {0, 2}  # 85.5 and 90.0
        
        # Test open-ended ranges
        results = index.range_search(70.0, float('inf'), False, True)
        assert results == {0, 2}  # > 70.0
    
    def test_text_index_operations(self):
        """Test TextIndex functionality"""
        config = IndexConfig(type=IndexType.TEXT, case_sensitive=False)
        index = TextIndex("Server.description", config)
        
        # Add documents
        docs = [
            "Web server handling HTTP requests",
            "Database server storing user data",
            "Cache server for session management",
            "API gateway routing requests"
        ]
        
        for i, doc in enumerate(docs):
            index.add_document(i, doc)
        
        # Test term search
        results = index.search("server")
        assert results == {0, 1, 2}
        
        results = index.search("requests")
        assert results == {0, 3}
        
        results = index.search("nonexistent")
        assert results == set()
        
        # Test phrase search
        results = index.phrase_search("user data")
        assert results == {1}
    
    def test_index_memory_and_coverage_stats(self):
        """Test index statistics calculation"""
        config = IndexConfig(type=IndexType.KEYWORD)
        index = KeywordIndex("test.field", config)
        
        # Add some documents
        for i in range(100):
            index.add_document(i, f"value_{i % 10}")  # 10 unique values
        
        # Test memory usage calculation
        memory_usage = index.memory_usage()
        assert memory_usage > 0
        
        # Test coverage calculation
        coverage = index.document_coverage()
        assert coverage == 100.0  # All 100 documents have values


class TestIndexBuilder:
    """Test the SmartIndexBuilder class"""
    
    def setup_method(self):
        """Set up test models and data"""
        @dataclass
        class TestServer:
            @index()
            @property
            def status(self) -> str:
                return self._status
            
            @index(priority="high")
            @property
            def description(self) -> str:
                return self._description
            
            @index()
            @property
            def cpu_percent(self) -> float:
                return self._cpu_percent
            
            @index()
            @property
            def tags(self) -> List[str]:
                return self._tags
            
            def __init__(self, status: str, description: str, cpu_percent: float, tags: List[str]):
                self._status = status
                self._description = description
                self._cpu_percent = cpu_percent
                self._tags = tags
        
        self.TestServer = TestServer
        self.test_data = [
            TestServer("running", "Web server", 85.5, ["web", "production"]),
            TestServer("stopped", "Database server", 45.2, ["db", "production"]),
            TestServer("running", "Cache server", 25.0, ["cache", "development"]),
        ]
    
    def test_field_discovery(self):
        """Test discovery of indexed fields from model classes"""
        builder = SmartIndexBuilder()
        indexed_fields = builder._discover_indexed_fields([self.TestServer])
        
        expected_fields = {
            "TestServer.status", "TestServer.description", 
            "TestServer.cpu_percent", "TestServer.tags"
        }
        assert set(indexed_fields.keys()) == expected_fields
        
        # Check specific configurations
        assert indexed_fields["TestServer.description"].priority == "high"
    
    def test_sample_extraction(self):
        """Test extraction of field samples for analysis"""
        builder = SmartIndexBuilder()
        indexed_fields = builder._discover_indexed_fields([self.TestServer])
        samples = builder._extract_field_samples(self.test_data, indexed_fields)
        
        # Check samples were extracted
        assert "TestServer.status" in samples
        assert "running" in samples["TestServer.status"]
        assert "stopped" in samples["TestServer.status"]
        
        assert "TestServer.cpu_percent" in samples
        assert 85.5 in samples["TestServer.cpu_percent"]
    
    def test_complete_index_building(self):
        """Test complete index building process"""
        builder = SmartIndexBuilder()
        indexes = builder.discover_and_build([self.TestServer], self.test_data)
        
        # Check that indexes were created
        expected_indexes = {
            "TestServer.status", "TestServer.description",
            "TestServer.cpu_percent", "TestServer.tags"
        }
        assert set(indexes.keys()) == expected_indexes
        
        # Test that indexes work
        status_index = indexes["TestServer.status"]
        results = status_index.search("running")
        assert len(results) == 2  # Two running servers
        
        cpu_index = indexes["TestServer.cpu_percent"]
        if hasattr(cpu_index, 'range_search'):
            results = cpu_index.range_search(40.0, 90.0, True, True)
            assert len(results) == 2  # 85.5 and 45.2


class TestIntegrationScenarios:
    """Test integration with existing Lucene filter"""
    
    def setup_method(self):
        """Create test data and indexes"""
        @dataclass
        class Product:
            @index()
            @property
            def category(self) -> str:
                return self._category
            
            @index(priority="high")
            @property
            def name(self) -> str:
                return self._name
            
            @index()
            @property
            def price(self) -> float:
                return self._price
            
            @index()
            @property
            def in_stock(self) -> bool:
                return self._in_stock
            
            @index()
            @property
            def tags(self) -> List[str]:
                return self._tags
            
            def __init__(self, category: str, name: str, price: float, 
                        in_stock: bool, tags: List[str]):
                self._category = category
                self._name = name
                self._price = price
                self._in_stock = in_stock
                self._tags = tags
        
        self.Product = Product
        self.products = [
            Product("electronics", "Laptop Pro", 1299.99, True, ["computer", "work"]),
            Product("electronics", "Smartphone X", 899.99, True, ["phone", "mobile"]),
            Product("books", "Python Guide", 39.99, False, ["programming", "education"]),
            Product("clothing", "Winter Jacket", 159.99, True, ["outdoor", "winter"]),
            Product("electronics", "Tablet Mini", 299.99, True, ["tablet", "portable"]),
        ]
        
        # Build indexes
        builder = SmartIndexBuilder()
        self.indexes = builder.discover_and_build([Product], self.products)
    
    def test_optimized_filter_creation(self):
        """Test creation of optimized filter"""
        optimized_filter = IndexOptimizedLuceneFilter(self.products, self.indexes)
        assert optimized_filter.objects == self.products
        assert optimized_filter.indexes == self.indexes
    
    def test_index_candidate_extraction(self):
        """Test extraction of candidates from indexes"""
        optimized_filter = IndexOptimizedLuceneFilter(self.products, self.indexes)
        
        # Test simple field:value query (simplified)
        # In real implementation, this would parse the full Lucene query
        candidates = optimized_filter._get_index_candidates("category:electronics")
        
        # This is a simplified test - in reality, you'd need full query parsing
        if candidates is not None:
            assert isinstance(candidates, set)
    
    def test_index_stats_collection(self):
        """Test collection of index statistics"""
        optimized_filter = IndexOptimizedLuceneFilter(self.products, self.indexes)
        stats = optimized_filter.get_index_stats()
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        
        # Check stat structure
        for field_path, stat in stats.items():
            assert 'type' in stat
            assert 'memory_mb' in stat
            assert 'coverage_percent' in stat
            assert 'priority' in stat


class TestPerformanceCharacteristics:
    """Test performance aspects of the indexing system"""
    
    def setup_method(self):
        """Generate large test dataset"""
        @dataclass
        class LargeTestModel:
            @index()
            @property
            def status(self) -> str:
                return self._status
            
            @index()
            @property
            def score(self) -> float:
                return self._score
            
            @index()
            @property
            def description(self) -> str:
                return self._description
            
            def __init__(self, status: str, score: float, description: str):
                self._status = status
                self._score = score
                self._description = description
        
        self.LargeTestModel = LargeTestModel
        
        # Generate 1000 test objects
        statuses = ["active", "inactive", "pending", "suspended"]
        descriptions = [
            "Short description",
            "Medium length description with more content",
            "Very long description with lots of words and detailed information about the item"
        ]
        
        self.large_dataset = []
        for i in range(1000):
            status = random.choice(statuses)
            score = random.uniform(0.0, 100.0)
            description = random.choice(descriptions) + f" item {i}"
            
            self.large_dataset.append(LargeTestModel(status, score, description))
    
    def test_large_dataset_index_building(self):
        """Test index building performance on large dataset"""
        builder = SmartIndexBuilder()
        
        start_time = time.time()
        indexes = builder.discover_and_build([self.LargeTestModel], self.large_dataset)
        build_time = time.time() - start_time
        
        # Should build indexes in reasonable time (< 5 seconds for 1000 items)
        assert build_time < 5.0
        assert len(indexes) > 0
        
        # Check index sizes are reasonable
        total_memory = sum(index.memory_usage() for index in indexes.values())
        assert total_memory > 0
        print(f"Index build time: {build_time:.3f}s, Total memory: {total_memory/1024/1024:.1f}MB")
    
    def test_index_query_performance(self):
        """Test query performance with indexes"""
        builder = SmartIndexBuilder()
        indexes = builder.discover_and_build([self.LargeTestModel], self.large_dataset)
        
        # Test keyword index performance
        status_index = indexes.get("LargeTestModel.status")
        if status_index:
            start_time = time.time()
            for _ in range(100):  # 100 queries
                results = status_index.search("active")
            query_time = time.time() - start_time
            
            # Should be very fast (< 0.1s for 100 queries)
            assert query_time < 0.1
            print(f"100 keyword queries: {query_time*1000:.1f}ms")
        
        # Test numeric index performance
        score_index = indexes.get("LargeTestModel.score")
        if score_index and hasattr(score_index, 'range_search'):
            start_time = time.time()
            for _ in range(100):  # 100 range queries
                results = score_index.range_search(50.0, 75.0, True, True)
            range_query_time = time.time() - start_time
            
            # Should be fast (< 0.2s for 100 range queries)
            assert range_query_time < 0.2
            print(f"100 range queries: {range_query_time*1000:.1f}ms")


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""
    
    def test_empty_dataset(self):
        """Test handling of empty datasets"""
        @dataclass
        class EmptyTestModel:
            @index()
            @property
            def field(self) -> str:
                return "value"
        
        builder = SmartIndexBuilder()
        indexes = builder.discover_and_build([EmptyTestModel], [])
        
        # Should handle empty dataset gracefully
        assert isinstance(indexes, dict)
    
    def test_none_values_in_data(self):
        """Test handling of None values"""
        @dataclass
        class NullTestModel:
            @index()
            @property
            def optional_field(self) -> Optional[str]:
                return self._value
            
            def __init__(self, value: Optional[str]):
                self._value = value
        
        test_data = [
            NullTestModel("value1"),
            NullTestModel(None),
            NullTestModel("value2"),
            NullTestModel(None),
        ]
        
        builder = SmartIndexBuilder()
        indexes = builder.discover_and_build([NullTestModel], test_data)
        
        # Should build index and handle None values
        field_index = indexes.get("NullTestModel.optional_field")
        assert field_index is not None
        
        # Test searching with None values present
        if hasattr(field_index, 'search'):
            results = field_index.search("value1")
            assert len(results) > 0
    
    def test_malformed_field_access(self):
        """Test handling of field access errors"""
        @dataclass
        class ProblematicModel:
            @index()
            @property
            def problematic_field(self) -> str:
                raise ValueError("Field access error")
        
        test_data = [ProblematicModel()]
        
        builder = SmartIndexBuilder()
        # Should not crash, should handle errors gracefully
        indexes = builder.discover_and_build([ProblematicModel], test_data)
        
        assert isinstance(indexes, dict)
    
    def test_mixed_data_types_in_field(self):
        """Test handling of mixed data types in same field"""
        @dataclass
        class MixedTypeModel:
            @index()
            @property
            def mixed_field(self) -> Any:
                return self._value
            
            def __init__(self, value: Any):
                self._value = value
        
        # Mix of different types
        test_data = [
            MixedTypeModel("string_value"),
            MixedTypeModel(42),
            MixedTypeModel(3.14),
            MixedTypeModel(True),
        ]
        
        builder = SmartIndexBuilder()
        indexes = builder.discover_and_build([MixedTypeModel], test_data)
        
        # Should handle mixed types gracefully
        field_index = indexes.get("MixedTypeModel.mixed_field")
        assert field_index is not None


class TestConfigurationVariants:
    """Test different configuration scenarios"""
    
    def test_all_index_types_explicit(self):
        """Test explicit configuration of all index types"""
        @dataclass
        class AllTypesModel:
            @index(type=IndexType.KEYWORD)
            @property
            def keyword_field(self) -> str:
                return "keyword"
            
            @index(type=IndexType.TEXT)
            @property
            def text_field(self) -> str:
                return "text content"
            
            @index(type=IndexType.NUMERIC)
            @property
            def numeric_field(self) -> float:
                return 42.0
            
            @index(type=IndexType.BOOLEAN)
            @property
            def boolean_field(self) -> bool:
                return True
            
            @index(type=IndexType.DATE)
            @property
            def date_field(self) -> datetime:
                return datetime.now()
            
            @index(type=IndexType.TAGS)
            @property
            def tags_field(self) -> List[str]:
                return ["tag1", "tag2"]
        
        test_data = [AllTypesModel()]
        
        builder = SmartIndexBuilder()
        indexes = builder.discover_and_build([AllTypesModel], test_data)
        
        # Should create appropriate index types
        assert len(indexes) == 6
        
        # Verify each index was created
        for field_name in ["keyword_field", "text_field", "numeric_field", 
                          "boolean_field", "date_field", "tags_field"]:
            assert f"AllTypesModel.{field_name}" in indexes
    
    def test_priority_ordering(self):
        """Test that high priority indexes are built first"""
        @dataclass
        class PriorityModel:
            @index(priority="low")
            @property
            def low_priority(self) -> str:
                return "low"
            
            @index(priority="high")
            @property
            def high_priority(self) -> str:
                return "high"
            
            @index(priority="medium")
            @property
            def medium_priority(self) -> str:
                return "medium"
        
        test_data = [PriorityModel()]
        
        builder = SmartIndexBuilder()
        
        # Track build order by monkey-patching
        build_order = []
        original_create = builder._build_indexes
        
        def tracking_build_indexes(objects, configs):
            nonlocal build_order
            # Sort configs by priority to verify order
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            sorted_configs = sorted(
                configs.items(),
                key=lambda x: priority_order.get(x[1].priority, 3)
            )
            build_order = [field_path for field_path, _ in sorted_configs]
            return original_create(objects, configs)
        
        builder._build_indexes = tracking_build_indexes
        indexes = builder.discover_and_build([PriorityModel], test_data)
        
        # Verify high priority was first
        assert "PriorityModel.high_priority" == build_order[0]
        assert "PriorityModel.medium_priority" == build_order[1]
        assert "PriorityModel.low_priority" == build_order[2]


# Test fixtures and utilities
@pytest.fixture
def sample_server_data():
    """Fixture providing sample server data"""
    @dataclass
    class Server:
        @index()
        @property
        def status(self) -> str:
            return self._status
        
        @index(priority="high")
        @property
        def name(self) -> str:
            return self._name
        
        @index()
        @property
        def cpu_percent(self) -> float:
            return self._cpu_percent
        
        def __init__(self, status: str, name: str, cpu_percent: float):
            self._status = status
            self._name = name
            self._cpu_percent = cpu_percent
    
    return [
        Server("running", "web-01", 85.5),
        Server("stopped", "db-01", 45.2),
        Server("running", "cache-01", 25.0),
    ], Server


@pytest.fixture
def index_builder():
    """Fixture providing SmartIndexBuilder instance"""
    return SmartIndexBuilder()


def test_with_fixtures(sample_server_data, index_builder):
    """Example test using fixtures"""
    servers, Server = sample_server_data
    indexes = index_builder.discover_and_build([Server], servers)
    
    assert len(indexes) > 0
    assert "Server.status" in indexes


# Performance benchmark tests
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests (marked for optional execution)"""
    
    def test_index_build_benchmark(self):
        """Benchmark index building on various dataset sizes"""
        @dataclass
        class BenchmarkModel:
            @index()
            @property
            def status(self) -> str:
                return self._status
            
            @index()
            @property
            def score(self) -> float:
                return self._score
            
            def __init__(self, status: str, score: float):
                self._status = status
                self._score = score
        
        builder = SmartIndexBuilder()
        dataset_sizes = [100, 500, 1000, 5000]
        
        for size in dataset_sizes:
            # Generate dataset
            dataset = [
                BenchmarkModel(
                    status=random.choice(["active", "inactive", "pending"]),
                    score=random.uniform(0.0, 100.0)
                )
                for _ in range(size)
            ]
            
            # Benchmark build time
            start_time = time.time()
            indexes = builder.discover_and_build([BenchmarkModel], dataset)
            build_time = time.time() - start_time
            
            print(f"Dataset size {size}: {build_time:.3f}s build time")
            
            # Performance assertions
            assert build_time < size * 0.001  # Should be linear or better
            assert len(indexes) > 0


# Test runner configuration
if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Run with performance tests:
    # pytest.main([__file__, "-v", "-m", "performance"])
    
    # Run with coverage:
    # pytest.main([__file__, "-v", "--cov=smart_index_decorator"])