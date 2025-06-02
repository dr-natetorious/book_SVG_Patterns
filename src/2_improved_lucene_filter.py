"""
Production-ready Lucene filtering with performance optimizations and comprehensive error handling
pip install luqum python-dateutil
"""

from luqum.parser import parser, ParseError
from luqum.visitor import TreeVisitor
from luqum.tree import (
    SearchField, Word, Phrase, Range, Fuzzy, Proximity, 
    AndOperation, OrOperation, NotOperation, Group,
    FieldGroup, UnknownOperation, Boost
)
from typing import Any, List, Dict, Union, Optional, Set
from weakref import WeakSet
from datetime import datetime, date
from dateutil.parser import parse as parse_date
import re
import time
import logging
import json
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class QueryMetrics:
    """Performance and execution metrics for queries"""
    query: str
    execution_time: float
    objects_processed: int
    cache_hits: int = 0
    cache_misses: int = 0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FilterConfig:
    """Configuration for the Lucene filter"""
    max_recursion_depth: int = 10
    max_field_count: int = 1000
    enable_text_cache: bool = True
    cache_size_limit: int = 10000
    enable_metrics: bool = True
    case_sensitive: bool = False
    fuzzy_threshold: float = 0.8
    date_formats: List[str] = field(default_factory=lambda: [
        '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
        '%d/%m/%Y', '%m/%d/%Y'
    ])


class TextExtractionCache:
    """LRU cache for extracted text from objects"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[int, List[str]] = {}
        self.access_order: List[int] = []
    
    def get(self, obj_id: int) -> Optional[List[str]]:
        """Get cached text for object"""
        if obj_id in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(obj_id)
            self.access_order.append(obj_id)
            return self.cache[obj_id]
        return None
    
    def put(self, obj_id: int, text_list: List[str]) -> None:
        """Store text in cache with LRU eviction"""
        if obj_id in self.cache:
            self.cache[obj_id] = text_list
            self.access_order.remove(obj_id)
            self.access_order.append(obj_id)
        else:
            # Evict least recently used if at capacity
            if len(self.cache) >= self.max_size:
                lru_id = self.access_order.pop(0)
                del self.cache[lru_id]
            
            self.cache[obj_id] = text_list
            self.access_order.append(obj_id)
    
    def clear(self) -> None:
        """Clear all cached data"""
        self.cache.clear()
        self.access_order.clear()


class QueryValidator:
    """Validates Lucene query syntax and structure"""
    
    @staticmethod
    def validate(query: str) -> Dict[str, Any]:
        """Validate query and return validation result"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'parsed_ast': None
        }
        
        if not query or not query.strip():
            result['errors'].append("Empty query not allowed")
            result['valid'] = False
            return result
        
        try:
            ast = parser.parse(query)
            result['parsed_ast'] = ast
            
            # Additional semantic validation
            QueryValidator._validate_semantic(ast, result)
            
        except ParseError as e:
            result['valid'] = False
            result['errors'].append(f"Parse error: {str(e)}")
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    @staticmethod
    def _validate_semantic(ast, result: Dict[str, Any]) -> None:
        """Perform semantic validation on parsed AST"""
        class ValidationVisitor(TreeVisitor):
            def visit_range(self, node: Range, context=None):
                # Validate range bounds
                try:
                    if node.low and node.high:
                        low_val = float(str(node.low))
                        high_val = float(str(node.high))
                        if low_val > high_val:
                            result['warnings'].append(f"Range has low > high: {low_val} > {high_val}")
                except ValueError:
                    pass  # Non-numeric ranges are valid
                return super().visit_range(node, context)
            
            def visit_search_field(self, node: SearchField, context=None):
                # Check for common field name issues
                field_name = str(node.name)
                if field_name.startswith('_'):
                    result['warnings'].append(f"Private field access: {field_name}")
                return super().visit_search_field(node, context)
        
        validator = ValidationVisitor()
        validator.visit(ast)


class ImprovedObjectFilterVisitor(TreeVisitor):
    """Enhanced visitor with performance optimizations and better error handling"""
    
    def __init__(self, obj: Any, config: FilterConfig, text_cache: TextExtractionCache):
        self.obj = obj
        self.config = config
        self.text_cache = text_cache
        self._visited: WeakSet = WeakSet()
        self._recursion_depth = 0
        self._field_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
    def visit_search_field(self, node: SearchField, context=None):
        """Handle field:value queries with enhanced error handling"""
        field_name = str(node.name)
        
        try:
            field_value = self._get_field_value(field_name)
            return self.visit(node.expr, context=field_value)
        except Exception as e:
            logging.debug(f"Field access error for {field_name}: {e}")
            return False
    
    def visit_word(self, node: Word, context=None):
        """Handle single word matches with improved type handling"""
        target_value = str(node.value)
        
        if context is not None:
            return self._compare_values(context, target_value)
        else:
            return self._fulltext_search(target_value)
    
    def visit_phrase(self, node: Phrase, context=None):
        """Handle quoted phrase matches"""
        target_phrase = str(node.value).strip('"\'')
        
        if context is not None:
            context_str = str(context)
            if not self.config.case_sensitive:
                return target_phrase.lower() in context_str.lower()
            return target_phrase in context_str
        else:
            return self._fulltext_search(target_phrase)
    
    def visit_range(self, node: Range, context=None):
        """Handle range queries with enhanced type conversion"""
        if context is None:
            return False
            
        try:
            field_val = self._convert_value(str(context))
            
            # Handle open-ended ranges
            if str(node.low) == '*':
                low_val = float('-inf')
                include_low = True
            else:
                low_val = self._convert_value(str(node.low))
                include_low = node.include_low
            
            if str(node.high) == '*':
                high_val = float('inf')
                include_high = True
            else:
                high_val = self._convert_value(str(node.high))
                include_high = node.include_high
            
            # Perform comparison
            if include_low:
                low_ok = field_val >= low_val
            else:
                low_ok = field_val > low_val
            
            if include_high:
                high_ok = field_val <= high_val
            else:
                high_ok = field_val < high_val
            
            return low_ok and high_ok
            
        except (ValueError, TypeError) as e:
            logging.debug(f"Range comparison error: {e}")
            return False
    
    def visit_fuzzy(self, node: Fuzzy, context=None):
        """Handle fuzzy matching with configurable threshold"""
        target = str(node.term)
        
        if context is not None:
            return self._fuzzy_match(str(context), target)
        return self._fulltext_fuzzy_search(target)
    
    def visit_proximity(self, node: Proximity, context=None):
        """Handle proximity queries (term1 NEAR term2)"""
        if context is None:
            return self._fulltext_proximity_search(node)
        
        # Simple proximity implementation
        context_str = str(context).lower()
        terms = [str(term).lower() for term in node.terms]
        distance = getattr(node, 'distance', 10)  # Default distance
        
        # Find all term positions
        positions = defaultdict(list)
        words = context_str.split()
        
        for i, word in enumerate(words):
            for term in terms:
                if term in word:
                    positions[term].append(i)
        
        # Check if all terms are within distance
        if len(positions) < len(terms):
            return False
        
        term_positions = list(positions.values())
        for pos1 in term_positions[0]:
            for other_positions in term_positions[1:]:
                if any(abs(pos1 - pos2) <= distance for pos2 in other_positions):
                    continue
                else:
                    return False
            return True
        
        return False
    
    def visit_boost(self, node: Boost, context=None):
        """Handle boost queries (term^2.0) - boost doesn't affect boolean logic"""
        return self.visit(node.expr, context)
    
    def visit_and_operation(self, node: AndOperation, context=None):
        """Handle AND operations with short-circuit evaluation"""
        for child in node.children:
            if not self.visit(child, context):
                return False
        return True
    
    def visit_or_operation(self, node: OrOperation, context=None):
        """Handle OR operations with short-circuit evaluation"""
        for child in node.children:
            if self.visit(child, context):
                return True
        return False
    
    def visit_not_operation(self, node: NotOperation, context=None):
        """Handle NOT operations"""
        return not self.visit(node.children[0], context)
    
    def visit_group(self, node: Group, context=None):
        """Handle parenthetical grouping"""
        return self.visit(node.children[0], context)
    
    def visit_field_group(self, node: FieldGroup, context=None):
        """Handle field:(expr1 expr2) grouping"""
        field_value = self._get_field_value(str(node.name))
        return self.visit(node.expr, context=field_value)
    
    def visit_unknown_operation(self, node: UnknownOperation, context=None):
        """Handle implicit operations with configurable default"""
        # Default to AND behavior for unknown operations
        results = [self.visit(child, context) for child in node.children]
        return all(results)
    
    def _get_field_value(self, field_path: str) -> Any:
        """Get field value with enhanced error handling and depth limits"""
        if self._field_count >= self.config.max_field_count:
            raise ValueError(f"Maximum field count exceeded: {self.config.max_field_count}")
        
        self._field_count += 1
        
        try:
            if self.obj in self._visited:
                return None
            
            if self._recursion_depth >= self.config.max_recursion_depth:
                return None
            
            self._visited.add(self.obj)
            self._recursion_depth += 1
            
            value = self.obj
            for part in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                elif isinstance(value, (list, tuple)) and part.isdigit():
                    idx = int(part)
                    value = value[idx] if 0 <= idx < len(value) else None
                else:
                    value = getattr(value, part, None)
                
                if value is None:
                    break
            
            self._recursion_depth -= 1
            return value
            
        except (KeyError, AttributeError, TypeError, IndexError, ValueError) as e:
            self._recursion_depth = max(0, self._recursion_depth - 1)
            logging.debug(f"Field access error for {field_path}: {e}")
            return None
    
    def _compare_values(self, field_value: Any, target_value: str) -> bool:
        """Enhanced value comparison with better type handling"""
        if field_value is None:
            return target_value.lower() in ['null', 'none', ''] if not self.config.case_sensitive else target_value in ['null', 'None', '']
        
        # Convert to strings for comparison
        field_str = str(field_value)
        target_str = target_value
        
        if not self.config.case_sensitive:
            field_str = field_str.lower()
            target_str = target_str.lower()
        
        # Handle wildcards with regex
        if '*' in target_str or '?' in target_str:
            pattern = target_str.replace('*', '.*').replace('?', '.')
            try:
                return bool(re.match(f'^{pattern}$', field_str))
            except re.error:
                return False
        
        # Exact match
        if field_str == target_str:
            return True
        
        # Numeric comparison
        try:
            field_num = self._convert_value(str(field_value))
            target_num = self._convert_value(target_value)
            return field_num == target_num
        except (ValueError, TypeError):
            pass
        
        # Date comparison
        try:
            field_date = self._parse_date(str(field_value))
            target_date = self._parse_date(target_value)
            if field_date and target_date:
                return field_date == target_date
        except (ValueError, TypeError):
            pass
        
        # Contains check as fallback
        return target_str in field_str
    
    def _convert_value(self, value: str) -> Union[int, float, bool, str]:
        """Enhanced type conversion with better error handling"""
        if not isinstance(value, str):
            return value
        
        value = value.strip('"\'')
        
        # Boolean
        if value.lower() in ('true', 'false', 'yes', 'no', '1', '0'):
            return value.lower() in ('true', 'yes', '1')
        
        # Integer
        try:
            if '.' not in value and 'e' not in value.lower() and value.lstrip('-+').isdigit():
                return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
        except ValueError:
            pass
        
        return value
    
    def _parse_date(self, value: str) -> Optional[datetime]:
        """Parse date strings using multiple formats"""
        # Try configured formats first
        for fmt in self.config.date_formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        
        # Try dateutil parser as fallback
        try:
            return parse_date(value)
        except (ValueError, TypeError):
            return None
    
    def _fulltext_search(self, term: str) -> bool:
        """Optimized full-text search with caching"""
        obj_id = id(self.obj)
        
        # Check cache first
        cached_text = None
        if self.config.enable_text_cache:
            cached_text = self.text_cache.get(obj_id)
        
        if cached_text is not None:
            self.cache_hits += 1
            all_text = cached_text
        else:
            self.cache_misses += 1
            all_text = self._extract_text_optimized(self.obj)
            if self.config.enable_text_cache:
                self.text_cache.put(obj_id, all_text)
        
        # Search in extracted text
        search_term = term if self.config.case_sensitive else term.lower()
        return any(search_term in text for text in all_text)
    
    def _extract_text_optimized(self, obj: Any, depth: int = 0) -> List[str]:
        """Optimized text extraction with depth and field limits"""
        if depth > self.config.max_recursion_depth:
            return []
        
        texts = []
        
        try:
            if isinstance(obj, str):
                text = obj if self.config.case_sensitive else obj.lower()
                texts.append(text)
            elif isinstance(obj, (int, float, bool)):
                texts.append(str(obj).lower())
            elif isinstance(obj, dict):
                for key, value in list(obj.items())[:100]:  # Limit dict size
                    texts.extend(self._extract_text_optimized(value, depth + 1))
            elif hasattr(obj, '__dict__'):
                for key, value in list(obj.__dict__.items())[:100]:  # Limit object attrs
                    texts.extend(self._extract_text_optimized(value, depth + 1))
            elif isinstance(obj, (list, tuple)):
                for item in list(obj)[:50]:  # Limit list size
                    texts.extend(self._extract_text_optimized(item, depth + 1))
            else:
                text = str(obj)
                if not self.config.case_sensitive:
                    text = text.lower()
                texts.append(text)
        except Exception as e:
            logging.debug(f"Text extraction error at depth {depth}: {e}")
        
        return texts
    
    def _fuzzy_match(self, text: str, target: str) -> bool:
        """Simple fuzzy matching implementation"""
        if not self.config.case_sensitive:
            text = text.lower()
            target = target.lower()
        
        # Simple fuzzy logic - can be enhanced with Levenshtein distance
        if target in text or text in target:
            return True
        
        # Character overlap fuzzy matching
        text_chars = set(text)
        target_chars = set(target)
        overlap = len(text_chars & target_chars)
        total = len(text_chars | target_chars)
        
        return (overlap / total) >= self.config.fuzzy_threshold if total > 0 else False
    
    def _fulltext_fuzzy_search(self, term: str) -> bool:
        """Fuzzy search across all text fields"""
        obj_id = id(self.obj)
        
        # Get cached or extract text
        cached_text = None
        if self.config.enable_text_cache:
            cached_text = self.text_cache.get(obj_id)
        
        if cached_text is not None:
            self.cache_hits += 1
            all_text = cached_text
        else:
            self.cache_misses += 1
            all_text = self._extract_text_optimized(self.obj)
            if self.config.enable_text_cache:
                self.text_cache.put(obj_id, all_text)
        
        # Fuzzy search in extracted text
        return any(self._fuzzy_match(text, term) for text in all_text)
    
    def _fulltext_proximity_search(self, node: Proximity) -> bool:
        """Proximity search across all text fields"""
        # Simplified proximity search - extract all text and search
        obj_id = id(self.obj)
        
        cached_text = None
        if self.config.enable_text_cache:
            cached_text = self.text_cache.get(obj_id)
        
        if cached_text is not None:
            self.cache_hits += 1
            all_text = ' '.join(cached_text)
        else:
            self.cache_misses += 1
            extracted = self._extract_text_optimized(self.obj)
            all_text = ' '.join(extracted)
            if self.config.enable_text_cache:
                self.text_cache.put(obj_id, extracted)
        
        # Use the same proximity logic as in visit_proximity
        return self.visit_proximity(node, context=all_text)


class EnhancedLuceneObjectFilter:
    """
    Production-ready Lucene-style filtering with comprehensive improvements:
    - Performance optimizations with caching
    - Enhanced error handling and validation
    - Comprehensive metrics and monitoring
    - Configurable behavior
    - Memory safety features
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self.text_cache = TextExtractionCache(self.config.cache_size_limit)
        self.metrics: List[QueryMetrics] = []
        self.logger = logging.getLogger(__name__)
    
    def filter(self, objects: List[Any], query: str) -> List[Any]:
        """Filter objects using enhanced Lucene query syntax"""
        start_time = time.time()
        
        # Validate query first
        validation = QueryValidator.validate(query)
        if not validation['valid']:
            error_msg = "; ".join(validation['errors'])
            self._record_metrics(query, start_time, 0, error=error_msg)
            self.logger.error(f"Query validation failed: {error_msg}")
            raise ValueError(f"Invalid query: {error_msg}")
        
        # Handle empty query
        if not query.strip():
            self._record_metrics(query, start_time, len(objects))
            return objects
        
        try:
            # Use pre-validated AST
            ast = validation['parsed_ast']
            
            results = []
            total_cache_hits = 0
            total_cache_misses = 0
            
            for obj in objects:
                visitor = ImprovedObjectFilterVisitor(obj, self.config, self.text_cache)
                
                try:
                    if visitor.visit(ast):
                        results.append(obj)
                    
                    total_cache_hits += visitor.cache_hits
                    total_cache_misses += visitor.cache_misses
                    
                except Exception as e:
                    self.logger.warning(f"Error processing object {id(obj)}: {e}")
                    continue
            
            # Record metrics
            execution_time = time.time() - start_time
            metrics = QueryMetrics(
                query=query,
                execution_time=execution_time,
                objects_processed=len(objects),
                cache_hits=total_cache_hits,
                cache_misses=total_cache_misses
            )
            
            if self.config.enable_metrics:
                self.metrics.append(metrics)
                # Keep only last 1000 metrics
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            self._record_metrics(query, start_time, len(objects), error=error_msg)
            self.logger.error(f"Filter execution error: {error_msg}")
            raise RuntimeError(f"Filter execution failed: {error_msg}")
    
    def explain_query(self, query: str) -> Dict[str, Any]:
        """Comprehensive query explanation with validation and structure"""
        validation = QueryValidator.validate(query)
        
        result = {
            'original_query': query,
            'validation': validation,
            'ast_structure': None,
            'estimated_complexity': 'unknown'
        }
        
        if validation['valid'] and validation['parsed_ast']:
            result['ast_structure'] = str(validation['parsed_ast'])
            result['estimated_complexity'] = self._estimate_complexity(validation['parsed_ast'])
        
        return result
    
    def get_metrics(self) -> List[QueryMetrics]:
        """Get query execution metrics"""
        return self.metrics.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.metrics:
            return {'message': 'No metrics available'}
        
        execution_times = [m.execution_time for m in self.metrics]
        cache_hit_rates = [
            m.cache_hits / (m.cache_hits + m.cache_misses) if (m.cache_hits + m.cache_misses) > 0 else 0
            for m in self.metrics
        ]
        
        return {
            'total_queries': len(self.metrics),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'avg_cache_hit_rate': sum(cache_hit_rates) / len(cache_hit_rates) if cache_hit_rates else 0,
            'errors': len([m for m in self.metrics if m.error]),
            'cache_size': len(self.text_cache.cache)
        }
    
    def clear_cache(self) -> None:
        """Clear text extraction cache"""
        self.text_cache.clear()
    
    def clear_metrics(self) -> None:
        """Clear collected metrics"""
        self.metrics.clear()
    
    def _record_metrics(self, query: str, start_time: float, objects_count: int, 
                       cache_hits: int = 0, cache_misses: int = 0, error: Optional[str] = None):
        """Record query metrics"""
        if self.config.enable_metrics:
            metrics = QueryMetrics(
                query=query,
                execution_time=time.time() - start_time,
                objects_processed=objects_count,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                error=error
            )
            self.metrics.append(metrics)
    
    def _estimate_complexity(self, ast) -> str:
        """Estimate query complexity based on AST structure"""
        class ComplexityVisitor(TreeVisitor):
            def __init__(self):
                self.complexity_score = 0
                self.depth = 0
                self.max_depth = 0
            
            def visit(self, node, context=None):
                self.depth += 1
                self.max_depth = max(self.max_depth, self.depth)
                
                # Add complexity based on node type
                if isinstance(node, (AndOperation, OrOperation)):
                    self.complexity_score += len(node.children)
                elif isinstance(node, Range):
                    self.complexity_score += 2
                elif isinstance(node, Fuzzy):
                    self.complexity_score += 3
                elif isinstance(node, Proximity):
                    self.complexity_score += 4
                else:
                    self.complexity_score += 1
                
                result = super().visit(node, context)
                self.depth -= 1
                return result
        
        visitor = ComplexityVisitor()
        visitor.visit(ast)
        
        score = visitor.complexity_score + visitor.max_depth
        
        if score <= 5:
            return 'low'
        elif score <= 15:
            return 'medium'
        elif score <= 30:
            return 'high'
        else:
            return 'very_high'


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test data
    servers = [
        {
            "name": "web-01",
            "cpu": 85.5,
            "memory": 16,
            "status": "running",
            "tags": ["web", "production"],
            "config": {"ssl": True, "port": 443},
            "created": "2024-01-15"
        },
        {
            "name": "db-primary",
            "cpu": 45.2,
            "memory": 32,
            "status": "running",
            "tags": ["database", "production"],
            "config": {"ssl": False, "port": 5432},
            "created": "2024-01-10"
        },
        {
            "name": "cache-dev",
            "cpu": 20.0,
            "memory": 8,
            "status": "stopped",
            "tags": ["cache", "development"],
            "config": {"ssl": False, "port": 6379},
            "created": "2024-01-20"
        }
    ]
    
    # Create filter with custom configuration
    config = FilterConfig(
        enable_text_cache=True,
        enable_metrics=True,
        case_sensitive=False
    )
    
    filter_engine = EnhancedLuceneObjectFilter(config)
    
    # Test queries
    test_queries = [
        "status:running",
        "cpu:[40.0 TO 90.0]",
        "name:*-01",
        "tags:production AND cpu:[60 TO *]",
        "config.ssl:true OR status:stopped",
        'name:"db-primary"',
        "NOT status:running",
        "(status:running OR status:idle) AND memory:[10 TO *]",
        "created:[2024-01-01 TO 2024-01-31]",
        "web~",  # Fuzzy search
    ]
    
    print("=== Enhanced Lucene Object Filter Demo ===")
    
    for query in test_queries:
        try:
            print(f"\nQuery: {query}")
            
            # Explain query
            explanation = filter_engine.explain_query(query)
            print(f"Complexity: {explanation['estimated_complexity']}")
            print(f"Valid: {explanation['validation']['valid']}")
            
            if explanation['validation']['warnings']:
                print(f"Warnings: {explanation['validation']['warnings']}")
            
            # Execute filter
            results = filter_engine.filter(servers, query)
            names = [s['name'] for s in results]
            print(f"Results: {names}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Performance summary
    print(f"\n=== Performance Summary ===")
    summary = filter_engine.get_performance_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")