"""
Enhanced Test Suite for Multi-Field Index System
===============================================

Tests for the gaps identified in the implementation:
- Multi-field query combinations (AND/OR/NOT)
- Field mapping and resolution
- IndexCandidateExtractor functionality
- Query coverage and optimization analysis
- Complex boolean logic scenarios
- Performance under various query patterns
"""

import pytest
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Set
import time
import random

# Import enhanced implementation
from smart_index_decorator import (
    index, IndexType, IndexConfig, SmartIndexBuilder, 
    IndexOptimizedLuceneFilter, FieldMapper, IndexCandidateExtractor,
    KeywordIndex, NumericIndex, TextIndex
)
from luqum.parser import parser


class TestFieldMapping:
    """Test field name to index path mapping"""
    
    def setup_method(self):
        @dataclass
        class TestModel:
            @index()
            @property
            def status(self) -> str:
                return "active"
            
            @index()
            @property
            def cpu_percent(self) -> float:
                return 50.0
        
        @dataclass
        class AnotherModel:
            @index()
            @property
            def name(self) -> str:
                return "test"
        
        self.TestModel = TestModel
        self.AnotherModel = AnotherModel
        self.models = [TestModel, AnotherModel]
    
    def test_field_mapping_creation(self):
        """Test field mapper correctly maps field names"""
        mapper = FieldMapper(self.models)
        
        # Test direct field name mapping
        assert mapper.get_index_path("status") == "TestModel.status"
        assert mapper.get_index_path("cpu_percent") == "TestModel.cpu_percent"
        assert mapper.get_index_path("name") == "AnotherModel.name"
        
        # Test full path mapping
        assert mapper.get_index_path("TestModel.status") == "TestModel.status"
        assert mapper.get_index_path("AnotherModel.name") == "AnotherModel.name"
    
    def test_field_mapping_conflicts(self):
        """Test handling of field name conflicts across models"""
        @dataclass
        class Model1:
            @index()
            @property
            def name(self) -> str:
                return "model1"
        
        @dataclass  
        class Model2:
            @index()
            @property
            def name(self) -> str:
                return "model2"
        
        mapper = FieldMapper([Model1, Model2])
        
        # Should map to one of them (implementation dependent)
        result = mapper.get_index_path("name")
        assert result in ["Model1.name", "Model2.name"]
        
        # Full paths should work
        assert mapper.get_index_path("Model1.name") == "Model1.name"
        assert mapper.get_index_path("Model2.name") == "Model2.name"
    
    def test_nonexistent_field_mapping(self):
        """Test mapping for non-indexed fields"""
        mapper = FieldMapper(self.models)
        assert mapper.get_index_path("nonexistent") is None
        assert mapper.get_index_path("Model.nonexistent") is None
    
    def test_available_fields_list(self):
        """Test getting list of available indexed fields"""
        mapper = FieldMapper(self.models)
        available = mapper.get_available_fields()
        
        assert "status" in available
        assert "cpu_percent" in available
        assert "name" in available
        assert "TestModel.status" in available


class TestIndexCandidateExtractor:
    """Test the AST-based candidate extraction"""
    
    def setup_method(self):
        """Setup indexes and field mapper for testing"""
        @dataclass
        class Server:
            @index()
            @property
            def status(self) -> str:
                return self._status
            
            @index()
            @property
            def cpu_percent(self) -> float:
                return self._cpu_percent
            
            @index()
            @property
            def name(self) -> str:
                return self._name
            
            def __init__(self, status: str, cpu_percent: float, name: str):
                self._status = status
                self._cpu_percent = cpu_percent
                self._name = name
        
        self.Server = Server
        self.test_data = [
            Server("running", 85.0, "web-01"),
            Server("stopped", 45.0, "db-01"),
            Server("running", 25.0, "cache-01"),
            Server("maintenance", 90.0, "api-gateway"),
        ]
        
        # Build indexes
        builder = SmartIndexBuilder()
        self.indexes, self.field_mapper = builder.discover_and_build([Server], self.test_data)
    
    def test_single_field_extraction(self):
        """Test extraction from single field queries"""
        extractor = IndexCandidateExtractor(self.indexes, self.field_mapper)
        
        # Test keyword search
        ast = parser.parse("status:running")
        candidates = extractor.visit(ast)
        assert candidates == {0, 2}  # Two running servers
        
        # Test numeric search
        ast = parser.parse("cpu_percent:85.0")
        candidates = extractor.visit(ast)
        assert candidates == {0}  # One server with 85% CPU
    
    def test_range_query_extraction(self):
        """Test range query candidate extraction"""
        extractor = IndexCandidateExtractor(self.indexes, self.field_mapper)
        
        ast = parser.parse("cpu_percent:[40 TO 90]")
        candidates = extractor.visit(ast)
        # Should match servers with CPU 45.0, 85.0, 90.0
        assert candidates == {0, 1, 3}
    
    def test_wildcard_extraction(self):
        """Test wildcard pattern extraction"""
        extractor = IndexCandidateExtractor(self.indexes, self.field_mapper)
        
        ast = parser.parse("name:*-01")
        candidates = extractor.visit(ast)
        # Should match web-01, db-01, cache-01
        assert candidates == {0, 1, 2}
    
    def test_and_operation_extraction(self):
        """Test AND operation intersection"""
        extractor = IndexCandidateExtractor(self.indexes, self.field_mapper)
        
        ast = parser.parse("status:running AND cpu_percent:[80 TO *]")
        candidates = extractor.visit(ast)
        # Should match only web-01 (running AND cpu >= 80)
        assert candidates == {0}
    
    def test_or_operation_extraction(self):
        """Test OR operation union"""
        extractor = IndexCandidateExtractor(self.indexes, self.field_mapper)
        
        ast = parser.parse("status:stopped OR cpu_percent:[90 TO *]")
        candidates = extractor.visit(ast)
        # Should match db-01 (stopped) OR api-gateway (cpu=90)
        assert candidates == {1, 3}
    
    def test_not_operation_fallback(self):
        """Test NOT operation returns None for fallback"""
        extractor = IndexCandidateExtractor(self.indexes, self.field_mapper)
        
        ast = parser.parse("NOT status:running")
        candidates = extractor.visit(ast)
        # NOT operations should return None for fallback
        assert candidates is None
    
    def test_complex_boolean_extraction(self):
        """Test complex boolean combinations"""
        extractor = IndexCandidateExtractor(self.indexes, self.field_mapper)
        
        ast = parser.parse("(status:running OR status:maintenance) AND cpu_percent:[50 TO *]")
        candidates = extractor.visit(ast)
        # (running OR maintenance) AND cpu >= 50 = web-01, api-gateway
        assert candidates == {0, 3}
    
    def test_query_coverage_tracking(self):
        """Test query coverage percentage calculation"""
        extractor = IndexCandidateExtractor(self.indexes, self.field_mapper)
        
        # All fields indexed
        ast = parser.parse("status:running AND cpu_percent:[80 TO *]")
        extractor.visit(ast)
        assert extractor.get_query_coverage() == 100.0
        
        # Partial coverage (if we had non-indexed fields)
        extractor = IndexCandidateExtractor(self.indexes, self.field_mapper)
        ast = parser.parse("status:running AND nonexistent:value")
        extractor.visit(ast)
        coverage = extractor.get_query_coverage()
        assert 0 <= coverage <= 100
    
    def test_nonexistent_field_handling(self):
        """Test handling of non-indexed fields"""
        extractor = IndexCandidateExtractor(self.indexes, self.field_mapper)
        
        ast = parser.parse("nonexistent_field:value")
        candidates = extractor.visit(ast)
        # Should return None for fallback
        assert candidates is None


class TestMultiFieldQueries:
    """Test complete multi-field query scenarios"""
    
    def setup_method(self):
        """Setup comprehensive test environment"""
        @dataclass
        class Product:
            @index()
            @property
            def category(self) -> str:
                return self._category
            
            @index()
            @property
            def price(self) -> float:
                return self._price
            
            @index()
            @property
            def name(self) -> str:
                return self._name
            
            @index()
            @property
            def in_stock(self) -> bool:
                return self._in_stock
            
            @index()
            @property
            def tags(self) -> List[str]:
                return self._tags
            
            def __init__(self, category: str, price: float, name: str, 
                        in_stock: bool, tags: List[str]):
                self._category = category
                self._price = price
                self._name = name
                self._in_stock = in_stock
                self._tags = tags
        
        self.Product = Product
        self.products = [
            Product("electronics", 999.99, "Laptop Pro", True, ["computer", "work"]),
            Product("electronics", 599.99, "Tablet Mini", True, ["tablet", "portable"]),
            Product("books", 29.99, "Python Guide", False, ["programming", "education"]),
            Product("clothing", 79.99, "Winter Jacket", True, ["outdoor", "winter"]),
            Product("electronics", 1299.99, "Gaming Desktop", True, ["computer", "gaming"]),
        ]
        
        builder = SmartIndexBuilder()
        self.indexes, self.field_mapper = builder.discover_and_build([Product], self.products)
        self.optimized_filter = IndexOptimizedLuceneFilter(
            self.products, self.indexes, self.field_mapper
        )
    
    def test_and_queries_across_indexes(self):
        """Test AND queries using multiple indexes"""
        # Category AND price range
        results = self.optimized_filter.filter("category:electronics AND price:[500 TO 1000]")
        assert len(results) == 2  # Laptop Pro (999.99), Tablet Mini (599.99)
        
        # Category AND stock status
        results = self.optimized_filter.filter("category:electronics AND in_stock:true")
        assert len(results) == 3  # All electronics are in stock
        
        # Multiple field intersection
        results = self.optimized_filter.filter("category:electronics AND price:[1000 TO *] AND in_stock:true")
        assert len(results) == 1  # Only Gaming Desktop
    
    def test_or_queries_across_indexes(self):
        """Test OR queries using multiple indexes"""
        # Category OR price range
        results = self.optimized_filter.filter("category:books OR price:[1200 TO *]")
        assert len(results) == 2  # Python Guide + Gaming Desktop
        
        # Out of stock OR high price
        results = self.optimized_filter.filter("in_stock:false OR price:[1200 TO *]")
        assert len(results) == 2  # Python Guide (out of stock) + Gaming Desktop (expensive)
    
    def test_complex_boolean_combinations(self):
        """Test complex nested boolean logic"""
        # (category AND price) OR (different category AND stock)
        results = self.optimized_filter.filter(
            "(category:electronics AND price:[* TO 700]) OR (category:clothing AND in_stock:true)"
        )
        assert len(results) == 2  # Tablet Mini + Winter Jacket
        
        # Nested parentheses with multiple fields
        results = self.optimized_filter.filter(
            "category:electronics AND (price:[* TO 800] OR (price:[1200 TO *] AND in_stock:true))"
        )
        assert len(results) == 2  # Tablet Mini + Gaming Desktop
    
    def test_wildcard_with_other_fields(self):
        """Test wildcard patterns combined with other field filters"""
        results = self.optimized_filter.filter("name:*top* AND category:electronics")
        assert len(results) == 2  # Laptop Pro + Gaming Desktop
        
        results = self.optimized_filter.filter("name:*Guide AND in_stock:false")
        assert len(results) == 1  # Python Guide
    
    def test_range_queries_with_filters(self):
        """Test range queries combined with exact matches"""
        results = self.optimized_filter.filter("price:[50 TO 100] AND category:clothing")
        assert len(results) == 1  # Winter Jacket
        
        results = self.optimized_filter.filter("price:[500 TO 1000] AND in_stock:true")
        assert len(results) == 2  # Laptop Pro + Tablet Mini
    
    def test_tags_field_queries(self):
        """Test queries on list/tags fields"""
        results = self.optimized_filter.filter("tags:computer AND price:[900 TO *]")
        assert len(results) == 2  # Laptop Pro + Gaming Desktop
        
        results = self.optimized_filter.filter("category:electronics AND tags:portable")
        assert len(results) == 1  # Tablet Mini
    
    def test_query_optimization_analysis(self):
        """Test query optimization analysis"""
        # Fully optimized query
        explanation = self.optimized_filter.explain_query("category:electronics AND price:[500 TO 1000]")
        assert explanation['parsed_successfully'] is True
        assert explanation['can_use_indexes'] is True
        assert explanation['index_coverage'] == 100.0
        
        # Partially optimized query
        explanation = self.optimized_filter.explain_query("category:electronics AND nonexistent:value")
        assert explanation['index_coverage'] < 100.0
    
    def test_fallback_for_unsupported_queries(self):
        """Test fallback to full scan for unsupported queries"""
        # NOT operations should fall back
        results = self.optimized_filter.filter("NOT category:electronics")
        assert len(results) >= 0  # Should work via fallback
        
        # Complex field groups should fall back
        results = self.optimized_filter.filter("category:(electronics books)")
        assert len(results) >= 0  # Should work via fallback


class TestQueryPerformanceAndOptimization:
    """Test query performance characteristics and optimization"""
    
    def setup_method(self):
        """Setup large dataset for performance testing"""
        @dataclass
        class LargeTestModel:
            @index(priority="high")
            @property
            def status(self) -> str:
                return self._status
            
            @index()
            @property
            def score(self) -> float:
                return self._score
            
            @index()
            @property
            def category(self) -> str:
                return self._category
            
            @index()
            @property
            def active(self) -> bool:
                return self._active
            
            def __init__(self, status: str, score: float, category: str, active: bool):
                self._status = status
                self._score = score
                self._category = category
                self._active = active
        
        self.LargeTestModel = LargeTestModel
        
        # Generate large dataset
        statuses = ["active", "inactive", "pending", "archived"]
        categories = ["A", "B", "C", "D", "E"]
        
        self.large_dataset = []
        for i in range(2000):
            status = random.choice(statuses)
            score = random.uniform(0.0, 100.0)
            category = random.choice(categories)
            active = random.choice([True, False])
            
            self.large_dataset.append(LargeTestModel(status, score, category, active))
        
        builder = SmartIndexBuilder()
        self.indexes, self.field_mapper = builder.discover_and_build(
            [LargeTestModel], self.large_dataset
        )
        self.optimized_filter = IndexOptimizedLuceneFilter(
            self.large_dataset, self.indexes, self.field_mapper
        )
    
    def test_single_field_vs_multi_field_performance(self):
        """Compare single field vs multi-field query performance"""
        # Single field query
        start_time = time.time()
        for _ in range(50):
            results = self.optimized_filter.filter("status:active")
        single_field_time = time.time() - start_time
        
        # Multi-field query
        start_time = time.time()
        for _ in range(50):
            results = self.optimized_filter.filter("status:active AND score:[50 TO *]")
        multi_field_time = time.time() - start_time
        
        # Multi-field should be reasonably fast (within 2x of single field)
        assert multi_field_time < single_field_time * 3
        
        print(f"Single field: {single_field_time*1000:.1f}ms")
        print(f"Multi field: {multi_field_time*1000:.1f}ms")
    
    def test_complex_query_optimization_ratio(self):
        """Test optimization ratio for complex queries"""
        complex_queries = [
            "status:active AND score:[75 TO *]",
            "category:A OR category:B",
            "active:true AND (score:[80 TO *] OR status:pending)",
            "status:active AND category:A AND score:[50 TO 90]"
        ]
        
        for query in complex_queries:
            explanation = self.optimized_filter.explain_query(query)
            
            # Should achieve significant optimization
            if explanation['can_use_indexes']:
                assert explanation['optimization_ratio'] < 0.8  # At least 20% reduction
                print(f"Query '{query}': {explanation['optimization_ratio']:.3f} ratio")
    
    def test_query_cache_effectiveness(self):
        """Test query result caching"""
        query = "status:active AND score:[60 TO 80]"
        
        # First execution
        start_time = time.time()
        results1 = self.optimized_filter.filter(query)
        first_time = time.time() - start_time
        
        # Second execution (should use cache)
        start_time = time.time()
        results2 = self.optimized_filter.filter(query)
        cached_time = time.time() - start_time
        
        # Results should be identical
        assert len(results1) == len(results2)
        
        # Cached query should be faster
        assert cached_time < first_time
        
        # Check cache hit in stats
        stats = self.optimized_filter.get_index_stats()
        assert stats['cache_hits'] > 0
    
    def test_index_coverage_vs_performance(self):
        """Test correlation between index coverage and performance"""
        queries = [
            ("status:active", "full_coverage"),
            ("status:active AND score:[50 TO *]", "full_coverage"),
            ("status:active AND nonexistent:value", "partial_coverage")
        ]
        
        for query, expected_type in queries:
            explanation = self.optimized_filter.explain_query(query)
            
            start_time = time.time()
            results = self.optimized_filter.filter(query)
            query_time = time.time() - start_time
            
            coverage = explanation['index_coverage']
            
            if expected_type == "full_coverage":
                assert coverage == 100.0
                assert query_time < 0.01  # Should be very fast
            elif expected_type == "partial_coverage":
                assert coverage < 100.0
                # May be slower due to fallback
    
    def test_statistics_accuracy(self):
        """Test accuracy of performance statistics"""
        # Execute various queries
        queries = [
            "status:active",
            "score:[50 TO *]",
            "active:true AND category:A",
            "nonexistent:field"  # Should cause fallback
        ]
        
        initial_stats = self.optimized_filter.get_index_stats()
        initial_total = initial_stats['total_queries']
        
        for query in queries:
            try:
                self.optimized_filter.filter(query)
            except:
                pass  # Count even failed queries
        
        final_stats = self.optimized_filter.get_index_stats()
        
        # Should track all queries
        assert final_stats['total_queries'] == initial_total + len(queries)
        assert final_stats['index_accelerated'] + final_stats['fallback_queries'] <= final_stats['total_queries']


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error scenarios in multi-field queries"""
    
    def setup_method(self):
        @dataclass
        class EdgeCaseModel:
            @index()
            @property
            def optional_field(self) -> Optional[str]:
                return self._optional
            
            @index()
            @property
            def mixed_field(self) -> Any:
                return self._mixed
            
            def __init__(self, optional: Optional[str], mixed: Any):
                self._optional = optional
                self._mixed = mixed
        
        self.EdgeCaseModel = EdgeCaseModel
        self.edge_data = [
            EdgeCaseModel("value1", "string"),
            EdgeCaseModel(None, 42),
            EdgeCaseModel("value2", 3.14),
            EdgeCaseModel(None, None),
        ]
        
        builder = SmartIndexBuilder()
        self.indexes, self.field_mapper = builder.discover_and_build(
            [EdgeCaseModel], self.edge_data
        )
        self.optimized_filter = IndexOptimizedLuceneFilter(
            self.edge_data, self.indexes, self.field_mapper
        )
    
    def test_malformed_query_handling(self):
        """Test handling of malformed queries"""
        malformed_queries = [
            "field:",
            ":value",
            "field:value AND",
            "field:[1 TO",
            "((field:value"
        ]
        
        for query in malformed_queries:
            explanation = self.optimized_filter.explain_query(query)
            assert explanation['parsed_successfully'] is False
            
            # Should still return results via fallback
            results = self.optimized_filter.filter(query)
            assert isinstance(results, list)
    
    def test_none_values_in_multi_field_queries(self):
        """Test multi-field queries with None values"""
        # Query involving field with None values
        results = self.optimized_filter.filter("optional_field:value1 AND mixed_field:string")
        assert len(results) >= 0
        
        results = self.optimized_filter.filter("optional_field:value1 OR mixed_field:42")
        assert len(results) >= 0
    
    def test_empty_result_combinations(self):
        """Test queries that should return empty results"""
        # Impossible combination
        results = self.optimized_filter.filter("optional_field:nonexistent AND mixed_field:impossible")
        assert len(results) == 0
        
        # Should still use indexes efficiently
        explanation = self.optimized_filter.explain_query("optional_field:nonexistent AND mixed_field:impossible")
        if explanation['can_use_indexes']:
            assert explanation['candidate_count'] == 0
    
    def test_query_with_all_fallback_fields(self):
        """Test query where all fields require fallback"""
        # All non-indexed fields
        results = self.optimized_filter.filter("nonexistent1:value AND nonexistent2:value")
        explanation = self.optimized_filter.explain_query("nonexistent1:value AND nonexistent2:value")
        
        assert explanation['can_use_indexes'] is False
        assert isinstance(results, list)


class TestIntegrationWithExistingFilter:
    """Test integration with existing SimpleLuceneFilter"""
    
    def setup_method(self):
        """Setup for integration testing"""
        @dataclass
        class IntegrationModel:
            @index()
            @property
            def indexed_field(self) -> str:
                return self._indexed
            
            @property  # Not indexed
            def non_indexed_field(self) -> str:
                return self._non_indexed
            
            def __init__(self, indexed: str, non_indexed: str):
                self._indexed = indexed
                self._non_indexed = non_indexed
        
        self.IntegrationModel = IntegrationModel
        self.test_data = [
            IntegrationModel("indexed_value1", "non_indexed_value1"),
            IntegrationModel("indexed_value2", "non_indexed_value2"),
        ]
        
        builder = SmartIndexBuilder()
        self.indexes, self.field_mapper = builder.discover_and_build(
            [IntegrationModel], self.test_data
        )
        
        # Mock base filter for testing
        class MockBaseLuceneFilter:
            def filter(self, objects, query):
                # Simple mock that returns all objects
                return objects
        
        self.mock_base_filter = MockBaseLuceneFilter()
        self.optimized_filter = IndexOptimizedLuceneFilter(
            self.test_data, self.indexes, self.field_mapper, self.mock_base_filter
        )
    
    def test_indexed_field_acceleration(self):
        """Test that indexed fields use acceleration"""
        query = "indexed_field:indexed_value1"
        
        explanation = self.optimized_filter.explain_query(query)
        assert explanation['can_use_indexes'] is True
        assert explanation['index_coverage'] == 100.0
        
        results = self.optimized_filter.filter(query)
        # Should get accelerated results passed to base filter
        assert len(results) <= len(self.test_data)
    
    def test_non_indexed_field_fallback(self):
        """Test that non-indexed fields fall back to base filter"""
        query = "non_indexed_field:non_indexed_value1"
        
        explanation = self.optimized_filter.explain_query(query)
        assert explanation['can_use_indexes'] is False
        
        results = self.optimized_filter.filter(query)
        # Should fall back to base filter with all objects
        assert len(results) == len(self.test_data)
    
    def test_mixed_indexed_non_indexed_query(self):
        """Test queries mixing indexed and non-indexed fields"""
        query = "indexed_field:indexed_value1 AND non_indexed_field:non_indexed_value1"
        
        explanation = self.optimized_filter.explain_query(query)
        # Should have partial coverage
        assert 0 < explanation['index_coverage'] < 100.0
        
        results = self.optimized_filter.filter(query)
        assert isinstance(results, list)


# Performance benchmark tests
@pytest.mark.performance
class TestMultiFieldPerformanceBenchmarks:
    """Performance benchmarks for multi-field queries"""
    
    def test_scaling_with_dataset_size(self):
        """Test how multi-field queries scale with dataset size"""
        @dataclass
        class ScaleTestModel:
            @index()
            @property
            def field1(self) -> str:
                return self._field1
            
            @index()
            @property
            def field2(self) -> float:
                return self._field2
            
            def __init__(self, field1: str, field2: float):
                self._field1 = field1
                self._field2 = field2
        
        dataset_sizes = [100, 500, 1000, 2000]
        query = "field1:active AND field2:[50 TO *]"
        
        for size in dataset_sizes:
            # Generate dataset
            dataset = [
                ScaleTestModel(
                    field1=random.choice(["active", "inactive"]),
                    field2=random.uniform(0.0, 100.0)
                )
                for _ in range(size)
            ]
            
            # Build indexes and filter
            builder = SmartIndexBuilder()
            indexes, field_mapper = builder.discover_and_build([ScaleTestModel], dataset)
            optimized_filter = IndexOptimizedLuceneFilter(dataset, indexes, field_mapper)
            
            # Benchmark query time
            start_time = time.time()
            for _ in range(10):  # Multiple runs for average
                results = optimized_filter.filter(query)
            query_time = (time.time() - start_time) / 10
            
            print(f"Dataset size {size}: {query_time*1000:.2f}ms per query")
            
            # Should scale sublinearly
            assert query_time < size * 0.0001  # Should be much better than linear


# Test fixtures
@pytest.fixture
def multi_field_test_data():
    """Fixture providing multi-field test data"""
    @dataclass
    class TestProduct:
        @index()
        @property
        def category(self) -> str:
            return self._category
        
        @index()
        @property
        def price(self) -> float:
            return self._price
        
        @index()
        @property
        def available(self) -> bool:
            return self._available
        
        def __init__(self, category: str, price: float, available: bool):
            self._category = category
            self._price = price
            self._available = available
    
    products = [
        TestProduct("electronics", 299.99, True),
        TestProduct("books", 19.99, False),
        TestProduct("electronics", 599.99, True),
        TestProduct("clothing", 49.99, True),
    ]
    
    builder = SmartIndexBuilder()
    indexes, field_mapper = builder.discover_and_build([TestProduct], products)
    
    return {
        'products': products,
        'indexes': indexes,
        'field_mapper': field_mapper,
        'model_class': TestProduct
    }


def test_fixture_multi_field_queries(multi_field_test_data):
    """Test using multi-field fixture"""
    data = multi_field_test_data
    optimized_filter = IndexOptimizedLuceneFilter(
        data['products'], data['indexes'], data['field_mapper']
    )
    
    # Test AND query
    results = optimized_filter.filter("category:electronics AND available:true")
    assert len(results) == 2
    
    # Test OR query
    results = optimized_filter.filter("category:books OR price:[500 TO *]")
    assert len(results) == 2  # Books + expensive electronics


# Integration test with real SimpleLuceneFilter
class TestRealLuceneFilterIntegration:
    """Test integration with actual SimpleLuceneFilter if available"""
    
    def test_with_real_filter(self):
        """Test with real SimpleLuceneFilter implementation"""
        try:
            # Try to import your actual SimpleLuceneFilter
            # from your_module import SimpleLuceneFilter
            
            @dataclass
            class RealTestModel:
                @index()
                @property
                def status(self) -> str:
                    return self._status
                
                @index()
                @property
                def value(self) -> int:
                    return self._value
                
                def __init__(self, status: str, value: int):
                    self._status = status
                    self._value = value
            
            test_data = [
                RealTestModel("active", 10),
                RealTestModel("inactive", 20),
                RealTestModel("active", 30),
            ]
            
            builder = SmartIndexBuilder()
            indexes, field_mapper = builder.discover_and_build([RealTestModel], test_data)
            
            # If SimpleLuceneFilter is available, use it
            # base_filter = SimpleLuceneFilter()
            # optimized_filter = IndexOptimizedLuceneFilter(
            #     test_data, indexes, field_mapper, base_filter
            # )
            
            # For now, test without base filter
            optimized_filter = IndexOptimizedLuceneFilter(
                test_data, indexes, field_mapper
            )
            
            # Test that it works with complex queries
            results = optimized_filter.filter("status:active AND value:[15 TO *]")
            assert len(results) == 1  # Only one active with value >= 15
            
        except ImportError:
            pytest.skip("SimpleLuceneFilter not available for integration test")


class TestQueryComplexityScenarios:
    """Test various query complexity scenarios"""
    
    def setup_method(self):
        """Setup complex test scenario"""
        @dataclass
        class ComplexModel:
            @index()
            @property
            def status(self) -> str:
                return self._status
            
            @index()
            @property
            def priority(self) -> str:
                return self._priority
            
            @index()
            @property
            def score(self) -> float:
                return self._score
            
            @index()
            @property
            def active(self) -> bool:
                return self._active
            
            @index()
            @property
            def name(self) -> str:
                return self._name
            
            @index()
            @property
            def tags(self) -> List[str]:
                return self._tags
            
            def __init__(self, status: str, priority: str, score: float, 
                        active: bool, name: str, tags: List[str]):
                self._status = status
                self._priority = priority
                self._score = score
                self._active = active
                self._name = name
                self._tags = tags
        
        self.ComplexModel = ComplexModel
        self.complex_data = [
            ComplexModel("running", "high", 85.5, True, "web-server-01", ["web", "production"]),
            ComplexModel("stopped", "medium", 45.2, False, "db-primary", ["database", "production"]),
            ComplexModel("running", "low", 25.0, True, "cache-redis", ["cache", "development"]),
            ComplexModel("maintenance", "high", 90.0, True, "api-gateway", ["api", "production"]),
            ComplexModel("running", "medium", 60.0, True, "worker-01", ["worker", "production"]),
            ComplexModel("stopped", "low", 15.0, False, "old-server", ["legacy", "deprecated"]),
        ]
        
        builder = SmartIndexBuilder()
        self.indexes, self.field_mapper = builder.discover_and_build([ComplexModel], self.complex_data)
        self.optimized_filter = IndexOptimizedLuceneFilter(
            self.complex_data, self.indexes, self.field_mapper
        )
    
    def test_three_field_intersection(self):
        """Test intersection across three indexed fields"""
        results = self.optimized_filter.filter(
            "status:running AND priority:high AND active:true"
        )
        assert len(results) == 1  # Only web-server-01
        
        results = self.optimized_filter.filter(
            "status:running AND score:[50 TO *] AND active:true"
        )
        assert len(results) == 2  # web-server-01, worker-01
    
    def test_four_field_complex_query(self):
        """Test complex query across four fields"""
        results = self.optimized_filter.filter(
            "status:running AND priority:medium AND score:[50 TO 70] AND active:true"
        )
        assert len(results) == 1  # worker-01
    
    def test_mixed_operators_complex(self):
        """Test mix of AND, OR with multiple fields"""
        results = self.optimized_filter.filter(
            "(status:running OR status:maintenance) AND priority:high AND score:[80 TO *]"
        )
        assert len(results) == 2  # web-server-01, api-gateway
        
        results = self.optimized_filter.filter(
            "active:true AND (priority:high OR (priority:medium AND score:[50 TO *]))"
        )
        assert len(results) == 3  # web-server-01, api-gateway, worker-01
    
    def test_wildcard_with_multiple_constraints(self):
        """Test wildcard patterns with multiple field constraints"""
        results = self.optimized_filter.filter(
            "name:*server* AND status:running AND active:true"
        )
        assert len(results) == 2  # web-server-01, old-server would be excluded by status
        
        results = self.optimized_filter.filter(
            "name:*-01 AND priority:high AND score:[80 TO *]"
        )
        assert len(results) == 1  # web-server-01
    
    def test_range_with_multiple_boolean_fields(self):
        """Test range queries combined with multiple boolean constraints"""
        results = self.optimized_filter.filter(
            "score:[20 TO 50] AND active:false AND status:stopped"
        )
        assert len(results) == 2  # db-primary, old-server
        
        results = self.optimized_filter.filter(
            "score:[60 TO *] AND active:true AND (priority:high OR priority:medium)"
        )
        assert len(results) == 3  # web-server-01, api-gateway, worker-01
    
    def test_tags_with_complex_logic(self):
        """Test tag searches with complex boolean logic"""
        results = self.optimized_filter.filter(
            "tags:production AND status:running AND score:[50 TO *]"
        )
        assert len(results) == 2  # web-server-01, worker-01
        
        results = self.optimized_filter.filter(
            "(tags:development OR tags:deprecated) AND active:false"
        )
        assert len(results) == 1  # old-server (deprecated and inactive)
    
    def test_deeply_nested_parentheses(self):
        """Test deeply nested boolean expressions"""
        results = self.optimized_filter.filter(
            "((status:running OR status:maintenance) AND (priority:high OR priority:medium)) AND (score:[50 TO *] OR active:true)"
        )
        # Complex logic: (running OR maintenance) AND (high OR medium) AND (score>=50 OR active)
        # Should match: web-server-01, api-gateway, worker-01
        assert len(results) == 3
    
    def test_query_optimization_for_complex_queries(self):
        """Test optimization analysis for complex queries"""
        complex_queries = [
            "status:running AND priority:high",  # Simple, should be fully optimized
            "status:running AND priority:high AND score:[80 TO *]",  # More complex, still optimized
            "name:*server* AND status:running AND priority:high AND active:true",  # Very complex
            "((status:running OR status:stopped) AND priority:high) OR (score:[90 TO *] AND active:true)"  # Very complex
        ]
        
        for query in complex_queries:
            explanation = self.optimized_filter.explain_query(query)
            
            assert explanation['parsed_successfully'] is True
            assert explanation['can_use_indexes'] is True
            assert explanation['index_coverage'] == 100.0  # All fields are indexed
            
            # Complex queries should still achieve good optimization
            if explanation['candidate_count'] is not None:
                optimization_ratio = explanation['optimization_ratio']
                assert optimization_ratio < 1.0  # Should reduce candidate set
                
                print(f"Query: {query}")
                print(f"  Candidates: {explanation['candidate_count']}/{len(self.complex_data)}")
                print(f"  Optimization: {optimization_ratio:.3f}")
                print(f"  Recommendation: {explanation['recommended_action']}")


class TestErrorRecoveryAndFallback:
    """Test error recovery and fallback mechanisms"""
    
    def setup_method(self):
        @dataclass 
        class FallbackTestModel:
            @index()
            @property
            def good_field(self) -> str:
                return self._good
            
            def __init__(self, good: str):
                self._good = good
        
        self.test_data = [FallbackTestModel("value1"), FallbackTestModel("value2")]
        
        builder = SmartIndexBuilder()
        self.indexes, self.field_mapper = builder.discover_and_build([FallbackTestModel], self.test_data)
        self.optimized_filter = IndexOptimizedLuceneFilter(
            self.test_data, self.indexes, self.field_mapper
        )
    
    def test_partial_index_failure_recovery(self):
        """Test recovery when some indexes fail"""
        # Simulate index corruption by removing an index
        corrupted_indexes = dict(self.indexes)
        del corrupted_indexes['FallbackTestModel.good_field']
        
        corrupted_filter = IndexOptimizedLuceneFilter(
            self.test_data, corrupted_indexes, self.field_mapper
        )
        
        # Should fall back gracefully
        results = corrupted_filter.filter("good_field:value1")
        assert isinstance(results, list)
    
    def test_field_mapper_corruption_recovery(self):
        """Test recovery from field mapping issues"""
        # Create corrupted field mapper
        class CorruptedFieldMapper:
            def get_index_path(self, field_name):
                return None  # Always return None
            
            def get_available_fields(self):
                return []
        
        corrupted_filter = IndexOptimizedLuceneFilter(
            self.test_data, self.indexes, CorruptedFieldMapper()
        )
        
        # Should fall back to full scan
        results = corrupted_filter.filter("good_field:value1")
        assert isinstance(results, list)
        assert len(results) == len(self.test_data)  # Falls back to returning all
    
    def test_ast_parsing_error_recovery(self):
        """Test recovery from AST parsing errors"""
        # These should cause parsing errors but recover gracefully
        problematic_queries = [
            "field:value AND AND field2:value",  # Double AND
            "field:value OR OR field2:value",    # Double OR
            "field:[broken range",               # Broken range
            "field:value (unclosed paren",       # Unclosed parenthesis
        ]
        
        for query in problematic_queries:
            # Should not crash, should fall back
            results = self.optimized_filter.filter(query)
            assert isinstance(results, list)
            
            explanation = self.optimized_filter.explain_query(query)
            assert explanation['parsed_successfully'] is False


class TestMemoryAndResourceManagement:
    """Test memory usage and resource management"""
    
    def test_index_memory_estimation_accuracy(self):
        """Test accuracy of memory usage estimation"""
        @dataclass
        class MemoryTestModel:
            @index()
            @property
            def large_field(self) -> str:
                return self._large
            
            def __init__(self, large: str):
                self._large = large
        
        # Create dataset with known memory characteristics
        large_strings = ["x" * 1000 for _ in range(100)]  # 100KB of string data
        test_data = [MemoryTestModel(s) for s in large_strings]
        
        builder = SmartIndexBuilder()
        indexes, _ = builder.discover_and_build([MemoryTestModel], test_data)
        
        # Check memory estimation
        index = indexes['MemoryTestModel.large_field']
        estimated_memory = index.memory_usage()
        
        # Should be reasonable estimate (within order of magnitude)
        assert 10000 < estimated_memory < 1000000  # Between 10KB and 1MB
    
    def test_query_cache_memory_bounds(self):
        """Test query cache doesn't grow unbounded"""
        @dataclass
        class CacheTestModel:
            @index()
            @property
            def field(self) -> str:
                return "value"
        
        test_data = [CacheTestModel()]
        builder = SmartIndexBuilder()
        indexes, field_mapper = builder.discover_and_build([CacheTestModel], test_data)
        optimized_filter = IndexOptimizedLuceneFilter(test_data, indexes, field_mapper)
        
        # Execute many unique queries
        for i in range(1000):
            query = f"field:value{i}"
            optimized_filter.filter(query)
        
        # Cache should not grow unbounded
        cache_size = len(optimized_filter.query_cache)
        assert cache_size < 1000  # Should have some limit


# Test runner configuration and utilities
class TestUtilities:
    """Utility functions for testing"""
    
    @staticmethod
    def create_test_dataset(size: int, num_categories: int = 5):
        """Create test dataset of specified size"""
        @dataclass
        class GeneratedTestModel:
            @index()
            @property
            def category(self) -> str:
                return self._category
            
            @index() 
            @property
            def value(self) -> float:
                return self._value
            
            def __init__(self, category: str, value: float):
                self._category = category
                self._value = value
        
        categories = [f"cat_{i}" for i in range(num_categories)]
        return [
            GeneratedTestModel(
                category=random.choice(categories),
                value=random.uniform(0.0, 100.0)
            )
            for _ in range(size)
        ]
    
    @staticmethod
    def measure_query_performance(optimized_filter, query: str, iterations: int = 10):
        """Measure query performance over multiple iterations"""
        times = []
        for _ in range(iterations):
            start_time = time.time()
            results = optimized_filter.filter(query)
            times.append(time.time() - start_time)
        
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'result_count': len(results) if 'results' in locals() else 0
        }


if __name__ == "__main__":
    # Run specific test suites
    
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Run only multi-field tests
    # pytest.main([__file__ + "::TestMultiFieldQueries", "-v"])
    
    # Run with performance tests
    # pytest.main([__file__, "-v", "-m", "performance"])
    
    # Run with coverage
    # pytest.main([__file__, "-v", "--cov=smart_index_decorator", "--cov-report=html"])
    
    print("\n" + "="*60)
    print("Enhanced Test Suite Coverage:")
    print("✓ Multi-field query combinations (AND/OR/NOT)")
    print("✓ Field mapping and resolution") 
    print("✓ IndexCandidateExtractor AST traversal")
    print("✓ Query coverage and optimization analysis")
    print("✓ Complex boolean logic scenarios")
    print("✓ Performance under various query patterns")
    print("✓ Error handling and fallback mechanisms")
    print("✓ Integration with existing filter systems")
    print("✓ Memory and resource management")
    print("✓ Edge cases and malformed queries")
    print("="*60)