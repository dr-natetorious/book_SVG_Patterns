"""
Smart Index Decorator System for Lucene Object Filter
=====================================================

Single @index decorator with intelligent defaults and type detection.
Integrates with existing SimpleLuceneFilter for massive performance gains.

Usage:
    @dataclass
    class Server:
        @index()  # Smart defaults based on field name and data
        def status(self) -> str: ...
        
        @index(priority="high", searchable=True)  # Override defaults
        def description(self) -> str: ...

Features:
    - Automatic type detection (TEXT, KEYWORD, NUMERIC, DATE, BOOLEAN, TAGS)
    - Smart defaults based on field names and data patterns
    - Minimal configuration required
    - High-performance index building for static data
    - Integration with existing Lucene filter classes
"""

from typing import Optional, Union, List, Dict, Any, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import re
import time
import bisect
from datetime import datetime, date


class IndexType(Enum):
    """Index type enumeration"""
    AUTO = "auto"           # Smart detection
    TEXT = "text"           # Full-text search
    KEYWORD = "keyword"     # Exact match
    NUMERIC = "numeric"     # Range queries  
    DATE = "date"           # Date ranges
    BOOLEAN = "boolean"     # True/false
    TAGS = "tags"          # List of keywords


@dataclass
class IndexConfig:
    """Index configuration with smart defaults"""
    enabled: bool = True
    type: IndexType = IndexType.AUTO
    priority: str = "medium"  # high, medium, low
    searchable: Optional[bool] = None
    filterable: Optional[bool] = None
    sortable: Optional[bool] = None
    case_sensitive: Optional[bool] = None
    options: Dict[str, Any] = field(default_factory=dict)
    
    # Resolved configuration (populated during analysis)
    resolved_type: Optional[IndexType] = None
    resolved_config: Dict[str, Any] = field(default_factory=dict)


def index(enabled: bool = True, 
          type: Optional[IndexType] = IndexType.AUTO,
          priority: str = "medium",
          searchable: Optional[bool] = None,
          filterable: Optional[bool] = None,
          sortable: Optional[bool] = None,
          case_sensitive: Optional[bool] = None,
          **kwargs) -> Callable:
    """
    Universal index decorator with smart defaults
    
    Args:
        enabled: Whether to index this field
        type: Index type (AUTO = smart detection)
        priority: Build priority (high/medium/low)
        searchable: Enable full-text search
        filterable: Enable exact filtering
        sortable: Enable range/sort queries
        case_sensitive: Case sensitivity (auto-detected)
        **kwargs: Additional index-specific options
    
    Examples:
        @index()  # Smart defaults
        def status(self) -> str: ...
        
        @index(priority="high", searchable=True)
        def description(self) -> str: ...
        
        @index(type=IndexType.NUMERIC, sortable=False)
        def score(self) -> float: ...
    """
    def decorator(func_or_property):
        config = IndexConfig(
            enabled=enabled,
            type=type,
            priority=priority,
            searchable=searchable,
            filterable=filterable,
            sortable=sortable,
            case_sensitive=case_sensitive,
            options=kwargs
        )
        
        # Attach configuration to property/method
        if isinstance(func_or_property, property):
            func_or_property.fget._index_config = config
        else:
            func_or_property._index_config = config
            
        return func_or_property
    return decorator


class SmartIndexAnalyzer:
    """Analyzes fields and applies intelligent defaults"""
    
    def analyze_field(self, field_name: str, sample_values: List[Any], 
                     config: IndexConfig) -> IndexConfig:
        """Analyze field and resolve smart defaults"""
        
        # Auto-detect type if AUTO
        if config.type == IndexType.AUTO:
            config.resolved_type = self._detect_field_type(field_name, sample_values)
        else:
            config.resolved_type = config.type
        
        # Apply smart defaults based on detected type
        self._apply_smart_defaults(config, field_name, sample_values)
        
        return config
    
    def _detect_field_type(self, field_name: str, sample_values: List[Any]) -> IndexType:
        """Smart field type detection based on name patterns and data"""
        if not sample_values:
            return IndexType.TEXT
        
        # Remove None values for analysis
        non_null_values = [v for v in sample_values if v is not None]
        if not non_null_values:
            return IndexType.TEXT
        
        first_value = non_null_values[0]
        
        # Type detection by Python type
        if isinstance(first_value, bool):
            return IndexType.BOOLEAN
        
        if isinstance(first_value, (int, float)):
            return IndexType.NUMERIC
        
        if isinstance(first_value, (datetime, date)):
            return IndexType.DATE
        
        if isinstance(first_value, (list, tuple)):
            return IndexType.TAGS
        
        if isinstance(first_value, str):
            return self._detect_string_type(field_name, non_null_values)
        
        return IndexType.TEXT
    
    def _detect_string_type(self, field_name: str, values: List[str]) -> IndexType:
        """Detect string field subtype based on patterns and characteristics"""
        
        # Common keyword field name patterns
        keyword_patterns = [
            r'.*status.*', r'.*state.*', r'.*type.*', r'.*category.*',
            r'.*environment.*', r'.*env.*', r'.*level.*', r'.*priority.*',
            r'.*severity.*', r'.*kind.*', r'.*role.*', r'.*group.*'
        ]
        
        # Check field name patterns
        field_lower = field_name.lower()
        for pattern in keyword_patterns:
            if re.match(pattern, field_lower):
                return IndexType.KEYWORD
        
        # Analyze value characteristics
        sample_size = min(100, len(values))
        sample_values = values[:sample_size]
        unique_values = set(sample_values)
        
        # Calculate metrics
        uniqueness_ratio = len(unique_values) / sample_size if sample_size > 0 else 0
        avg_length = sum(len(str(v)) for v in sample_values[:20]) / min(20, len(sample_values))
        
        # Decision logic
        if uniqueness_ratio < 0.1:  # Very few unique values (enum-like)
            return IndexType.KEYWORD
        elif avg_length > 50:  # Long text content
            return IndexType.TEXT
        elif uniqueness_ratio > 0.8 and avg_length > 10:  # High uniqueness, medium length
            return IndexType.TEXT
        else:
            return IndexType.KEYWORD
    
    def _apply_smart_defaults(self, config: IndexConfig, field_name: str, 
                            sample_values: List[Any]) -> None:
        """Apply intelligent defaults based on field type and characteristics"""
        
        # Smart defaults by index type
        type_defaults = {
            IndexType.TEXT: {
                'searchable': True,
                'filterable': False,
                'sortable': False,
                'case_sensitive': False
            },
            IndexType.KEYWORD: {
                'searchable': False,
                'filterable': True,
                'sortable': True,
                'case_sensitive': True
            },
            IndexType.NUMERIC: {
                'searchable': False,
                'filterable': True,
                'sortable': True,
                'case_sensitive': None
            },
            IndexType.DATE: {
                'searchable': False,
                'filterable': True,
                'sortable': True,
                'case_sensitive': None
            },
            IndexType.BOOLEAN: {
                'searchable': False,
                'filterable': True,
                'sortable': False,
                'case_sensitive': None
            },
            IndexType.TAGS: {
                'searchable': True,
                'filterable': True,
                'sortable': False,
                'case_sensitive': False
            }
        }
        
        defaults = type_defaults.get(config.resolved_type, {})
        
        # Apply defaults only if not explicitly set
        for key, default_value in defaults.items():
            if getattr(config, key) is None:
                setattr(config, key, default_value)
        
        # Store resolved configuration
        config.resolved_config = {
            'type': config.resolved_type,
            'searchable': config.searchable,
            'filterable': config.filterable,
            'sortable': config.sortable,
            'case_sensitive': config.case_sensitive,
            **config.options
        }


class BaseIndex:
    """Base class for all index implementations"""
    
    def __init__(self, field_path: str, config: IndexConfig):
        self.field_path = field_path
        self.config = config
        self.doc_count = 0
        
    def add_document(self, doc_id: int, value: Any) -> None:
        raise NotImplementedError
        
    def search(self, query: Any) -> Set[int]:
        raise NotImplementedError
        
    def memory_usage(self) -> int:
        """Return approximate memory usage in bytes"""
        return 0
    
    def document_coverage(self) -> float:
        """Return percentage of documents that have this field"""
        return 0.0


class KeywordIndex(BaseIndex):
    """Hash-based index for exact keyword matching"""
    
    def __init__(self, field_path: str, config: IndexConfig):
        super().__init__(field_path, config)
        self.value_to_docs: Dict[str, Set[int]] = defaultdict(set)
        self.total_docs = 0
        
    def add_document(self, doc_id: int, value: Any) -> None:
        if value is not None:
            key = str(value)
            if not self.config.case_sensitive:
                key = key.lower()
            self.value_to_docs[key].add(doc_id)
            self.doc_count += 1
        self.total_docs += 1
    
    def search(self, query: str) -> Set[int]:
        key = str(query)
        if not self.config.case_sensitive:
            key = key.lower()
        return self.value_to_docs.get(key, set())
    
    def wildcard_search(self, pattern: str) -> Set[int]:
        """Support wildcard patterns like status:running*"""
        if not self.config.case_sensitive:
            pattern = pattern.lower()
            
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        try:
            regex = re.compile(f'^{regex_pattern}$')
            results = set()
            for key, doc_ids in self.value_to_docs.items():
                if regex.match(key):
                    results.update(doc_ids)
            return results
        except re.error:
            return set()
    
    def memory_usage(self) -> int:
        return sum(len(key) * 4 + len(docs) * 8 
                  for key, docs in self.value_to_docs.items())
    
    def document_coverage(self) -> float:
        return (self.doc_count / self.total_docs * 100) if self.total_docs > 0 else 0


class NumericIndex(BaseIndex):
    """Sorted array index for numeric range queries"""
    
    def __init__(self, field_path: str, config: IndexConfig):
        super().__init__(field_path, config)
        self.sorted_values: List[tuple] = []  # [(value, doc_id), ...]
        self.built = False
        self.temp_values: List[tuple] = []
        
    def add_document(self, doc_id: int, value: Any) -> None:
        if value is not None:
            try:
                numeric_value = float(value)
                self.temp_values.append((numeric_value, doc_id))
                self.doc_count += 1
            except (ValueError, TypeError):
                pass  # Skip non-numeric values
    
    def build(self) -> None:
        """Build sorted index after all documents added"""
        if not self.built:
            self.sorted_values = sorted(self.temp_values)
            self.temp_values = []  # Free memory
            self.built = True
    
    def range_search(self, min_val: float, max_val: float, 
                    include_min: bool = True, include_max: bool = True) -> Set[int]:
        """Binary search for range queries"""
        if not self.built:
            self.build()
            
        # Find start and end positions
        if include_min:
            start_idx = bisect.bisect_left(self.sorted_values, (min_val, -1))
        else:
            start_idx = bisect.bisect_right(self.sorted_values, (min_val, float('inf')))
            
        if include_max:
            end_idx = bisect.bisect_right(self.sorted_values, (max_val, float('inf')))
        else:
            end_idx = bisect.bisect_left(self.sorted_values, (max_val, -1))
        
        return {doc_id for _, doc_id in self.sorted_values[start_idx:end_idx]}
    
    def search(self, query: Union[float, str]) -> Set[int]:
        """Exact value search"""
        try:
            value = float(query)
            return self.range_search(value, value, True, True)
        except (ValueError, TypeError):
            return set()
    
    def memory_usage(self) -> int:
        return len(self.sorted_values) * 16  # 8 bytes float + 8 bytes int
    
    def document_coverage(self) -> float:
        return (self.doc_count / len(self.sorted_values) * 100) if self.sorted_values else 0


class TextIndex(BaseIndex):
    """Inverted index for full-text search"""
    
    def __init__(self, field_path: str, config: IndexConfig):
        super().__init__(field_path, config)
        self.term_to_docs: Dict[str, Set[int]] = defaultdict(set)
        self.total_docs = 0
        
    def add_document(self, doc_id: int, value: Any) -> None:
        if value is not None:
            text = str(value)
            if not self.config.case_sensitive:
                text = text.lower()
            
            # Simple tokenization
            terms = self._tokenize(text)
            for term in terms:
                self.term_to_docs[term].add(doc_id)
            
            if terms:  # Only count if we extracted terms
                self.doc_count += 1
        self.total_docs += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization"""
        # Remove punctuation and split on whitespace
        clean_text = re.sub(r'[^\w\s]', ' ', text)
        return [term for term in clean_text.split() if len(term) > 2]
    
    def search(self, query: str) -> Set[int]:
        """Search for term"""
        if not self.config.case_sensitive:
            query = query.lower()
        return self.term_to_docs.get(query, set())
    
    def phrase_search(self, phrase: str) -> Set[int]:
        """Simple phrase search (term intersection)"""
        terms = self._tokenize(phrase)
        if not terms:
            return set()
        
        result = self.search(terms[0])
        for term in terms[1:]:
            result &= self.search(term)
        return result
    
    def memory_usage(self) -> int:
        return sum(len(term) * 4 + len(docs) * 8 
                  for term, docs in self.term_to_docs.items())
    
    def document_coverage(self) -> float:
        return (self.doc_count / self.total_docs * 100) if self.total_docs > 0 else 0


class SmartIndexBuilder:
    """Builds optimized indexes with smart analysis"""
    
    def __init__(self):
        self.analyzer = SmartIndexAnalyzer()
        self.index_builders = {
            IndexType.KEYWORD: KeywordIndex,
            IndexType.NUMERIC: NumericIndex,
            IndexType.TEXT: TextIndex,
            IndexType.BOOLEAN: KeywordIndex,  # Booleans use keyword index
            IndexType.DATE: NumericIndex,     # Dates use numeric index
            IndexType.TAGS: TextIndex,        # Tags use text index
        }
    
    def discover_and_build(self, model_classes: List[type], 
                          objects: List[Any]) -> Dict[str, BaseIndex]:
        """Discover fields and build optimized indexes"""
        
        print("🔍 Discovering indexed fields...")
        indexed_fields = self._discover_indexed_fields(model_classes)
        print(f"   Found {len(indexed_fields)} indexed fields")
        
        print("📊 Analyzing field samples...")
        field_samples = self._extract_field_samples(objects, indexed_fields)
        
        print("⚙️  Resolving smart configurations...")
        resolved_configs = {}
        for field_path, config in indexed_fields.items():
            samples = field_samples.get(field_path, [])
            field_name = field_path.split('.')[-1]
            
            resolved_config = self.analyzer.analyze_field(
                field_name, samples, config
            )
            resolved_configs[field_path] = resolved_config
        
        print("🏗️  Building indexes...")
        indexes = self._build_indexes(objects, resolved_configs)
        
        print("📈 Index build complete!")
        self._print_index_summary(resolved_configs, indexes)
        
        return indexes
    
    def _discover_indexed_fields(self, model_classes: List[type]) -> Dict[str, IndexConfig]:
        """Find all @index decorated fields"""
        indexed_fields = {}
        
        for cls in model_classes:
            for attr_name in dir(cls):
                attr = getattr(cls, attr_name)
                
                if isinstance(attr, property) and hasattr(attr.fget, '_index_config'):
                    config = attr.fget._index_config
                    if config.enabled:
                        field_path = f"{cls.__name__}.{attr_name}"
                        indexed_fields[field_path] = config
        
        return indexed_fields
    
    def _extract_field_samples(self, objects: List[Any], 
                             indexed_fields: Dict[str, IndexConfig]) -> Dict[str, List[Any]]:
        """Extract sample values for smart analysis"""
        samples = {field_path: [] for field_path in indexed_fields.keys()}
        
        # Sample up to 100 objects for analysis
        sample_objects = objects[:100]
        
        for obj in sample_objects:
            for field_path in indexed_fields.keys():
                try:
                    value = self._get_field_value(obj, field_path)
                    if value is not None:
                        samples[field_path].append(value)
                except Exception:
                    continue  # Skip problematic fields
        
        return samples
    
    def _get_field_value(self, obj: Any, field_path: str) -> Any:
        """Extract field value from object"""
        class_name, field_name = field_path.split('.', 1)
        
        if obj.__class__.__name__ == class_name:
            return getattr(obj, field_name, None)
        
        return None
    
    def _build_indexes(self, objects: List[Any], 
                      resolved_configs: Dict[str, IndexConfig]) -> Dict[str, BaseIndex]:
        """Build indexes based on resolved configurations"""
        indexes = {}
        
        # Sort by priority for build order
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_configs = sorted(
            resolved_configs.items(),
            key=lambda x: priority_order.get(x[1].priority, 3)
        )
        
        for field_path, config in sorted_configs:
            if config.resolved_type in self.index_builders:
                print(f"   Building {config.resolved_type.value} index for {field_path}")
                
                # Create index
                index_class = self.index_builders[config.resolved_type]
                index = index_class(field_path, config)
                
                # Populate index
                for doc_id, obj in enumerate(objects):
                    value = self._get_field_value(obj, field_path)
                    index.add_document(doc_id, value)
                
                # Finalize index (for sorted indexes)
                if hasattr(index, 'build'):
                    index.build()
                
                indexes[field_path] = index
        
        return indexes
    
    def _print_index_summary(self, resolved_configs: Dict[str, IndexConfig], 
                           indexes: Dict[str, BaseIndex]) -> None:
        """Print summary of built indexes"""
        print("\n📊 Smart Index Configuration Summary")
        print("=" * 60)
        
        by_type = defaultdict(list)
        total_memory = 0
        
        for field_path, config in resolved_configs.items():
            index_type = config.resolved_type.value
            index = indexes.get(field_path)
            
            memory_mb = 0
            coverage = 0
            if index:
                memory_mb = index.memory_usage() / (1024 * 1024)
                coverage = index.document_coverage()
                total_memory += memory_mb
            
            by_type[index_type].append((field_path, config, memory_mb, coverage))
        
        for index_type, fields in by_type.items():
            print(f"\n🔍 {index_type.upper()} Fields:")
            for field_path, config, memory_mb, coverage in fields:
                capabilities = []
                if config.searchable:
                    capabilities.append("searchable")
                if config.filterable:
                    capabilities.append("filterable")
                if config.sortable:
                    capabilities.append("sortable")
                
                caps_str = ", ".join(capabilities) if capabilities else "index-only"
                priority_emoji = "🔥" if config.priority == "high" else "⚡" if config.priority == "medium" else "📋"
                
                print(f"  {priority_emoji} {field_path}: {caps_str}")
                print(f"      Memory: {memory_mb:.1f}MB, Coverage: {coverage:.1f}%")
        
        print(f"\n💾 Total Index Memory: {total_memory:.1f}MB")


class IndexOptimizedLuceneFilter:
    """Lucene filter that leverages pre-built indexes for performance"""
    
    def __init__(self, objects: List[Any], indexes: Dict[str, BaseIndex]):
        self.objects = objects
        self.indexes = indexes
        # Import your existing SimpleLuceneFilter here
        # self.base_filter = SimpleLuceneFilter()
        self.query_cache = {}
    
    def filter(self, query: str) -> List[Any]:
        """Filter using indexes for massive speedup"""
        if query in self.query_cache:
            candidate_ids = self.query_cache[query]
        else:
            candidate_ids = self._get_index_candidates(query)
            self.query_cache[query] = candidate_ids
        
        # Convert doc IDs to objects
        if candidate_ids is not None:
            # Fast path: use index candidates
            candidate_objects = [self.objects[i] for i in candidate_ids 
                               if i < len(self.objects)]
            
            # Apply full Lucene filter to candidates only
            # return self.base_filter.filter(candidate_objects, query)
            return candidate_objects  # Simplified for demo
        else:
            # Fallback: full scan with existing filter
            # return self.base_filter.filter(self.objects, query)
            return self.objects  # Simplified for demo
    
    def _get_index_candidates(self, query: str) -> Optional[Set[int]]:
        """Extract candidate document IDs from indexes"""
        # This would integrate with your existing luqum parser
        # to extract indexable query parts and map them to indexes
        
        # Simplified example:
        if ":" in query:
            # Simple field:value query
            parts = query.split(":", 1)
            if len(parts) == 2:
                field_path = f"Server.{parts[0]}"  # Would need proper class mapping
                if field_path in self.indexes:
                    return self.indexes[field_path].search(parts[1])
        
        return None  # Fallback to full scan
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about index usage"""
        stats = {}
        for field_path, index in self.indexes.items():
            stats[field_path] = {
                'type': index.config.resolved_type.value,
                'memory_mb': index.memory_usage() / (1024 * 1024),
                'coverage_percent': index.document_coverage(),
                'priority': index.config.priority
            }
        return stats


# Example usage
if __name__ == "__main__":
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List
    
    @dataclass
    class Server:
        """Example server model with smart indexing"""
        
        @index()  # Smart defaults: KEYWORD, filterable=True
        @property 
        def status(self) -> str:
            return self._status
        
        @index(priority="high")  # Smart defaults: TEXT, searchable=True  
        @property
        def description(self) -> str:
            return self._description
        
        @index()  # Smart defaults: NUMERIC, sortable=True
        @property
        def cpu_percent(self) -> float:
            return self._cpu_percent
        
        @index(searchable=True)  # Override: make numeric searchable too
        @property  
        def memory_gb(self) -> int:
            return self._memory_gb
        
        @index()  # Smart defaults: TAGS, searchable=True, filterable=True
        @property
        def tags(self) -> List[str]:
            return self._tags
        
        @index(priority="low")  # Smart defaults: DATE, sortable=True
        @property
        def created_at(self) -> datetime:
            return self._created_at
        
        # Non-indexed field - no decorator
        @property
        def internal_id(self) -> str:
            return self._internal_id
        
        def __init__(self, status: str, description: str, cpu_percent: float,
                     memory_gb: int, tags: List[str]):
            self._status = status
            self._description = description
            self._cpu_percent = cpu_percent
            self._memory_gb = memory_gb
            self._tags = tags
            self._created_at = datetime.now()
            self._internal_id = f"srv_{id(self)}"
    
    # Example application startup
    def initialize_search_system():
        """Example of how to initialize the search system"""
        
        # Sample data
        servers = [
            Server("running", "Web server handling HTTP requests", 85.5, 16, ["web", "production"]),
            Server("stopped", "Database server for user data", 45.2, 32, ["database", "production"]),
            Server("running", "Cache server for session storage", 25.0, 8, ["cache", "development"]),
            Server("maintenance", "API gateway for microservices", 60.8, 12, ["api", "production"]),
        ]
        
        # Build indexes with smart analysis
        builder = SmartIndexBuilder()
        indexes = builder.discover_and_build([Server], servers)
        
        # Create optimized filter
        optimized_filter = IndexOptimizedLuceneFilter(servers, indexes)
        
        # Example queries
        print("\n🔍 Example Queries:")
        queries = ["status:running", "memory_gb:16", "production"]
        
        for query in queries:
            results = optimized_filter.filter(query)
            print(f"  Query '{query}': {len(results)} results")
        
        return optimized_filter
    
    # Run example
    if __name__ == "__main__":
        search_system = initialize_search_system()