"""
Index Candidate Extractor - Multi-Field Query AST Processing
===========================================================

Standalone module for extracting index candidates from Lucene queries.
Implements proper luqum AST traversal for complex multi-field queries.

Handles:
- Single field queries: status:running
- Range queries: cpu:[40 TO 90] 
- Wildcard queries: name:web*
- Boolean combinations: status:running AND cpu:[80 TO *]
- Complex nested logic: (status:running OR status:stopped) AND priority:high
"""

from typing import Dict, Set, Any, Optional
from luqum.visitor import TreeVisitor
from luqum.tree import (
    SearchField, Word, Phrase, Range, Fuzzy, Proximity,
    AndOperation, OrOperation, NotOperation, Group, FieldGroup,
    UnknownOperation, Boost
)


class IndexCandidateExtractor(TreeVisitor):
    """
    Extracts candidate document IDs from indexes using luqum AST traversal.
    
    Optimizes multi-field queries by:
    1. Identifying indexable query parts
    2. Extracting candidates from relevant indexes
    3. Combining results using boolean logic
    4. Tracking query coverage for optimization analysis
    """
    
    def __init__(self, indexes: Dict[str, Any], field_mapper: Any):
        """
        Initialize extractor with indexes and field mapping.
        
        Args:
            indexes: Dict mapping field paths to index instances
            field_mapper: FieldMapper instance for resolving field names
        """
        self.indexes = indexes
        self.field_mapper = field_mapper
        self.query_coverage = 0.0
        self.total_clauses = 0
        self.indexed_clauses = 0
        self.debug_log = []
    
    def visit_search_field(self, node: SearchField, context=None):
        """
        Handle field:value queries with comprehensive index lookup.
        
        Supports:
        - Exact matches: status:running
        - Wildcard patterns: name:web*
        - Range queries: cpu:[40 TO 90]
        - Phrase queries: description:"exact phrase"
        - Fuzzy queries: name:test~
        """
        self.total_clauses += 1
        field_name = str(node.name)
        field_path = self.field_mapper.get_index_path(field_name)
        
        self.debug_log.append(f"Processing field: {field_name} -> {field_path}")
        
        if field_path not in self.indexes:
            self.debug_log.append(f"No index found for {field_path}")
            return None
            
        index = self.indexes[field_path]
        self.indexed_clauses += 1
        
        try:
            if isinstance(node.expr, Word):
                # Handle word queries with wildcard support
                word_value = str(node.expr.value)
                self.debug_log.append(f"Word query: {word_value}")
                
                if self._has_wildcards(word_value):
                    if hasattr(index, 'wildcard_search'):
                        result = index.wildcard_search(word_value)
                        self.debug_log.append(f"Wildcard search returned {len(result)} candidates")
                        return result
                    else:
                        self.debug_log.append(f"Index {field_path} doesn't support wildcards")
                        return None
                else:
                    result = index.search(word_value)
                    self.debug_log.append(f"Exact search returned {len(result)} candidates")
                    return result
                
            elif isinstance(node.expr, Range):
                # Handle range queries with proper bounds
                if hasattr(index, 'range_search'):
                    min_val = self._convert_range_value(str(node.expr.low))
                    max_val = self._convert_range_value(str(node.expr.high))
                    
                    self.debug_log.append(
                        f"Range query: [{min_val} TO {max_val}] "
                        f"(include_low={node.expr.include_low}, include_high={node.expr.include_high})"
                    )
                    
                    result = index.range_search(
                        min_val, max_val,
                        node.expr.include_low, node.expr.include_high
                    )
                    self.debug_log.append(f"Range search returned {len(result)} candidates")
                    return result
                else:
                    self.debug_log.append(f"Index {field_path} doesn't support range queries")
                    return None
                    
            elif isinstance(node.expr, Phrase):
                # Handle phrase queries
                phrase_value = str(node.expr.value).strip('"\'')
                self.debug_log.append(f"Phrase query: '{phrase_value}'")
                
                if hasattr(index, 'phrase_search'):
                    result = index.phrase_search(phrase_value)
                    self.debug_log.append(f"Phrase search returned {len(result)} candidates")
                    return result
                else:
                    # Fallback to regular search
                    result = index.search(phrase_value)
                    self.debug_log.append(f"Phrase fallback search returned {len(result)} candidates")
                    return result
                
            elif isinstance(node.expr, Fuzzy):
                # Handle fuzzy queries
                fuzzy_term = str(node.expr.term)
                self.debug_log.append(f"Fuzzy query: {fuzzy_term}~")
                
                if hasattr(index, 'fuzzy_search'):
                    result = index.fuzzy_search(fuzzy_term)
                    self.debug_log.append(f"Fuzzy search returned {len(result)} candidates")
                    return result
                else:
                    # Fallback to exact search
                    result = index.search(fuzzy_term)
                    self.debug_log.append(f"Fuzzy fallback search returned {len(result)} candidates")
                    return result
                
            elif isinstance(node.expr, Proximity):
                # Handle proximity queries (rare but supported)
                self.debug_log.append("Proximity query - falling back")
                return None
                
        except Exception as e:
            self.debug_log.append(f"Error processing field {field_name}: {e}")
            # Index search failed, return None for fallback
            pass
        
        return None
    
    def visit_and_operation(self, node: AndOperation, context=None):
        """
        Handle AND operations using set intersection.
        
        Optimizes by:
        1. Processing all child nodes
        2. Collecting non-None candidate sets
        3. Intersecting all sets for final result
        """
        self.debug_log.append(f"Processing AND operation with {len(node.children)} children")
        
        candidate_sets = []
        for i, child in enumerate(node.children):
            candidates = self.visit(child, context)
            if candidates is not None:
                candidate_sets.append(candidates)
                self.debug_log.append(f"AND child {i}: {len(candidates)} candidates")
            else:
                self.debug_log.append(f"AND child {i}: no index candidates (fallback needed)")
        
        if candidate_sets:
            result = set.intersection(*candidate_sets)
            self.debug_log.append(f"AND intersection: {len(result)} final candidates")
            return result
        else:
            self.debug_log.append("AND operation: no indexable children")
            return None
    
    def visit_or_operation(self, node: OrOperation, context=None):
        """
        Handle OR operations using set union.
        
        Optimizes by:
        1. Processing all child nodes
        2. Collecting non-None candidate sets
        3. Taking union of all sets for final result
        """
        self.debug_log.append(f"Processing OR operation with {len(node.children)} children")
        
        candidate_sets = []
        for i, child in enumerate(node.children):
            candidates = self.visit(child, context)
            if candidates is not None:
                candidate_sets.append(candidates)
                self.debug_log.append(f"OR child {i}: {len(candidates)} candidates")
            else:
                self.debug_log.append(f"OR child {i}: no index candidates")
        
        if candidate_sets:
            result = set.union(*candidate_sets)
            self.debug_log.append(f"OR union: {len(result)} final candidates")
            return result
        else:
            self.debug_log.append("OR operation: no indexable children")
            return None
    
    def visit_not_operation(self, node: NotOperation, context=None):
        """
        Handle NOT operations.
        
        NOT operations require knowledge of all documents to exclude matches,
        so we return None to trigger fallback to full search.
        """
        self.debug_log.append("NOT operation detected - requires fallback")
        return None
    
    def visit_group(self, node: Group, context=None):
        """Handle parenthetical grouping by processing inner expression."""
        self.debug_log.append("Processing grouped expression")
        return self.visit(node.children[0], context)
    
    def visit_field_group(self, node: FieldGroup, context=None):
        """
        Handle field:(expr1 expr2) grouping.
        
        This is complex because it applies a field prefix to multiple expressions.
        For now, we fall back to full search for safety.
        """
        self.debug_log.append(f"Field group detected for {node.name} - requires fallback")
        return None
    
    def visit_unknown_operation(self, node: UnknownOperation, context=None):
        """
        Handle implicit operations (usually implicit AND).
        
        luqum parses "term1 term2" as UnknownOperation, which typically
        means implicit AND in Lucene syntax.
        """
        self.debug_log.append(f"Unknown operation with {len(node.children)} children (treating as AND)")
        
        candidate_sets = []
        for i, child in enumerate(node.children):
            candidates = self.visit(child, context)
            if candidates is not None:
                candidate_sets.append(candidates)
                self.debug_log.append(f"Unknown child {i}: {len(candidates)} candidates")
        
        if candidate_sets:
            result = set.intersection(*candidate_sets)
            self.debug_log.append(f"Unknown operation intersection: {len(result)} final candidates")
            return result
        else:
            self.debug_log.append("Unknown operation: no indexable children")
            return None
    
    def visit_boost(self, node: Boost, context=None):
        """Handle boost operations by processing the base expression."""
        self.debug_log.append(f"Boost operation (factor={node.force}) - processing base expression")
        return self.visit(node.children[0], context)
    
    def generic_visit(self, node, context=None):
        """
        Handle any unrecognized node types.
        
        This provides safety for new luqum node types or edge cases.
        """
        self.debug_log.append(f"Unrecognized node type: {type(node).__name__} - falling back")
        return None
    
    def _has_wildcards(self, value: str) -> bool:
        """Check if a value contains wildcard characters."""
        return '*' in value or '?' in value
    
    def _convert_range_value(self, value: str) -> float:
        """
        Convert luqum range values to numeric.
        
        Handles:
        - Wildcard bounds: * -> +/- infinity
        - Numeric values: "42" -> 42.0
        - Invalid values: fallback to 0.0
        """
        if value == '*':
            return float('inf')
        elif value == '-*':
            return float('-inf')
        
        try:
            return float(value)
        except (ValueError, TypeError):
            self.debug_log.append(f"Invalid range value '{value}', using 0.0")
            return 0.0
    
    def get_query_coverage(self) -> float:
        """
        Calculate percentage of query clauses that could use indexes.
        
        Returns:
            Float percentage (0.0 to 100.0) of query coverage
        """
        if self.total_clauses == 0:
            return 0.0
        return (self.indexed_clauses / self.total_clauses) * 100
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """
        Get detailed optimization metrics for query analysis.
        
        Returns:
            Dict with coverage stats, clause counts, and debug info
        """
        return {
            'total_clauses': self.total_clauses,
            'indexed_clauses': self.indexed_clauses,
            'coverage_percent': self.get_query_coverage(),
            'optimization_ratio': self.indexed_clauses / max(1, self.total_clauses),
            'debug_log': self.debug_log.copy(),
            'indexes_used': len(set(
                log.split('->')[1].strip() for log in self.debug_log 
                if '->' in log and 'No index found' not in log
            ))
        }
    
    def clear_debug_log(self) -> None:
        """Clear debug log for next query."""
        self.debug_log.clear()
        self.total_clauses = 0
        self.indexed_clauses = 0
        self.query_coverage = 0.0


class QueryOptimizationAnalyzer:
    """
    Helper class for analyzing query optimization potential.
    
    Provides recommendations for improving query performance through indexing.
    """
    
    @staticmethod
    def analyze_query_optimization(extractor: IndexCandidateExtractor, 
                                 candidates: Optional[Set[int]], 
                                 total_objects: int) -> Dict[str, Any]:
        """
        Analyze optimization effectiveness and provide recommendations.
        
        Args:
            extractor: IndexCandidateExtractor that processed the query
            candidates: Set of candidate document IDs (or None)
            total_objects: Total number of objects in dataset
            
        Returns:
            Dict with optimization analysis and recommendations
        """
        metrics = extractor.get_optimization_metrics()
        
        if candidates is None:
            return {
                **metrics,
                'can_use_indexes': False,
                'candidate_count': None,
                'reduction_ratio': 1.0,
                'recommendation': 'Consider adding indexes for fields in this query',
                'recommendation_priority': 'high'
            }
        
        reduction_ratio = len(candidates) / max(1, total_objects)
        coverage = metrics['coverage_percent']
        
        # Generate recommendation based on performance characteristics
        if coverage == 100 and reduction_ratio < 0.1:
            recommendation = "Excellent: Fully optimized with high selectivity"
            priority = "none"
        elif coverage > 80 and reduction_ratio < 0.3:
            recommendation = "Good: Well optimized"
            priority = "low"
        elif coverage > 50:
            recommendation = "Partial: Some fields indexed, consider indexing remaining fields"
            priority = "medium"
        elif coverage > 0:
            recommendation = "Poor: Limited index usage, add indexes for better performance"
            priority = "high"
        else:
            recommendation = "No optimization: No indexed fields found"
            priority = "critical"
        
        return {
            **metrics,
            'can_use_indexes': True,
            'candidate_count': len(candidates),
            'reduction_ratio': reduction_ratio,
            'selectivity': 1.0 - reduction_ratio,
            'recommendation': recommendation,
            'recommendation_priority': priority,
            'estimated_speedup': max(1.0, 1.0 / reduction_ratio) if reduction_ratio > 0 else 1.0
        }


# Example usage and testing
if __name__ == "__main__":
    """
    Example demonstrating IndexCandidateExtractor usage.
    This would typically be called from IndexOptimizedLuceneFilter.
    """
    
    # Mock data for testing
    class MockIndex:
        def __init__(self, data):
            self.data = data
            
        def search(self, query):
            return {i for i, val in enumerate(self.data) if str(val).lower() == query.lower()}
            
        def wildcard_search(self, pattern):
            import re
            regex = re.compile(pattern.replace('*', '.*').replace('?', '.'))
            return {i for i, val in enumerate(self.data) if regex.match(str(val).lower())}
            
        def range_search(self, min_val, max_val, include_min, include_max):
            results = set()
            for i, val in enumerate(self.data):
                try:
                    num_val = float(val)
                    if include_min and include_max:
                        if min_val <= num_val <= max_val:
                            results.add(i)
                    # Add other boundary conditions as needed
                except:
                    pass
            return results
    
    class MockFieldMapper:
        def get_index_path(self, field_name):
            return f"Test.{field_name}" if field_name in ['status', 'cpu', 'name'] else None
    
    # Mock indexes
    indexes = {
        'Test.status': MockIndex(['running', 'stopped', 'running', 'maintenance']),
        'Test.cpu': MockIndex([85.5, 45.2, 90.0, 25.0]),
        'Test.name': MockIndex(['web-server-01