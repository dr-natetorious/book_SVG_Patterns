"""
Simple Lucene Object Filter leveraging native luqum functionality
Focuses on the 80/20 rule for filtering Python objects with Lucene syntax
pip install luqum
"""

from luqum.parser import parser
from luqum.visitor import TreeTransformer
from luqum.utils import UnknownOperationResolver
from luqum.tree import (
    SearchField, Word, Phrase, Range, Fuzzy, 
    AndOperation, OrOperation, NotOperation, Group, FieldGroup
)
from typing import Any, List, Dict, Union, Optional
import re
from datetime import datetime


class ObjectFilterTransformer(TreeTransformer):
    """
    Transforms luqum AST into object filtering logic using native TreeTransformer.
    Leverages luqum's built-in capabilities instead of reinventing the wheel.
    """
    
    def __init__(self, target_object: Any):
        super().__init__()
        self.target_object = target_object
        self.current_context = None
    
    def visit_search_field(self, node: SearchField, context=None):
        """Handle field:value queries using luqum's native field handling"""
        field_name = str(node.name)
        field_value = self._get_field_value(field_name)
        
        # Transform the expression with field value as context
        result = self.visit(node.expr, context=field_value)
        
        # Return boolean result, not transformed node
        return result
    
    def visit_word(self, node: Word, context=None):
        """Handle word matches - core 80% case"""
        target = str(node.value)
        
        if context is not None:
            # Field-specific search
            return self._match_value(context, target)
        else:
            # Full-text search across object
            return self._fulltext_search(target)
    
    def visit_phrase(self, node: Phrase, context=None):
        """Handle quoted phrases"""
        target = str(node.value).strip('"\'')
        
        if context is not None:
            return target.lower() in str(context).lower()
        else:
            return self._fulltext_search(target)
    
    def visit_range(self, node: Range, context=None):
        """Handle range queries [min TO max]"""
        if context is None:
            return False
        
        try:
            value = self._convert_to_number(context)
            low = self._convert_to_number(str(node.low)) if str(node.low) != '*' else float('-inf')
            high = self._convert_to_number(str(node.high)) if str(node.high) != '*' else float('inf')
            
            if node.include_low:
                low_ok = value >= low
            else:
                low_ok = value > low
                
            if node.include_high:
                high_ok = value <= high
            else:
                high_ok = value < high
                
            return low_ok and high_ok
            
        except (ValueError, TypeError):
            # Fallback to string comparison for non-numeric ranges
            str_value = str(context).lower()
            str_low = str(node.low).lower() if str(node.low) != '*' else ''
            str_high = str(node.high).lower() if str(node.high) != '*' else 'zzz'
            
            return str_low <= str_value <= str_high
    
    def visit_fuzzy(self, node: Fuzzy, context=None):
        """Handle fuzzy matching with ~"""
        target = str(node.term)
        
        if context is not None:
            return self._fuzzy_match(str(context), target)
        else:
            return self._fulltext_fuzzy_search(target)
    
    def visit_and_operation(self, node: AndOperation, context=None):
        """Handle AND operations - use luqum's native structure"""
        for child in node.children:
            if not self.visit(child, context):
                return False
        return True
    
    def visit_or_operation(self, node: OrOperation, context=None):
        """Handle OR operations"""
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
        """Handle field:(expr1 expr2) grouping using luqum's native support"""
        field_value = self._get_field_value(str(node.name))
        return self.visit(node.expr, context=field_value)
    
    def _get_field_value(self, field_path: str) -> Any:
        """Extract field value using dot notation - core functionality"""
        try:
            value = self.target_object
            for part in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                elif isinstance(value, (list, tuple)) and part.isdigit():
                    idx = int(part)
                    value = value[idx] if 0 <= idx < len(value) else None
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return None
                
                if value is None:
                    break
            
            return value
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
    
    def _match_value(self, field_value: Any, target: str) -> bool:
        """Core value matching logic - handles 80% of cases"""
        if field_value is None:
            return target.lower() in ['null', 'none', '']
        
        field_str = str(field_value).lower()
        target_str = target.lower()
        
        # Wildcard matching
        if '*' in target_str or '?' in target_str:
            pattern = target_str.replace('*', '.*').replace('?', '.')
            try:
                return bool(re.match(f'^{pattern}$', field_str))
            except re.error:
                return False
        
        # Exact match
        if field_str == target_str:
            return True
        
        # Boolean matching
        if target_str in ['true', 'false']:
            try:
                bool_val = str(field_value).lower() in ['true', '1', 'yes', 'on']
                return bool_val == (target_str == 'true')
            except:
                pass
        
        # Numeric matching
        try:
            return float(field_value) == float(target)
        except (ValueError, TypeError):
            pass
        
        # Contains matching as fallback
        return target_str in field_str
    
    def _convert_to_number(self, value: Any) -> float:
        """Convert value to number for range comparisons"""
        if isinstance(value, (int, float)):
            return float(value)
        
        # Try to parse as number
        str_val = str(value).strip()
        if str_val.lower() in ['true', 'yes', '1']:
            return 1.0
        elif str_val.lower() in ['false', 'no', '0']:
            return 0.0
        
        return float(str_val)
    
    def _fulltext_search(self, term: str) -> bool:
        """Simple full-text search across object"""
        term_lower = term.lower()
        
        def extract_strings(obj, depth=0):
            if depth > 5:  # Simple recursion limit
                return []
            
            strings = []
            try:
                if isinstance(obj, str):
                    strings.append(obj.lower())
                elif isinstance(obj, dict):
                    for v in obj.values():
                        strings.extend(extract_strings(v, depth + 1))
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        strings.extend(extract_strings(item, depth + 1))
                elif hasattr(obj, '__dict__'):
                    for v in obj.__dict__.values():
                        strings.extend(extract_strings(v, depth + 1))
                else:
                    strings.append(str(obj).lower())
            except:
                pass  # Skip problematic objects
            
            return strings
        
        all_text = extract_strings(self.target_object)
        return any(term_lower in text for text in all_text)
    
    def _fuzzy_match(self, text: str, target: str) -> bool:
        """Simple fuzzy matching - contains or character overlap"""
        text_lower = text.lower()
        target_lower = target.lower()
        
        # Simple fuzzy logic
        if target_lower in text_lower or text_lower in target_lower:
            return True
        
        # Character overlap
        text_chars = set(text_lower)
        target_chars = set(target_lower)
        overlap = len(text_chars & target_chars)
        total = len(text_chars | target_chars)
        
        return (overlap / total) > 0.6 if total > 0 else False
    
    def _fulltext_fuzzy_search(self, term: str) -> bool:
        """Fuzzy search across all text"""
        def extract_strings(obj, depth=0):
            if depth > 5:
                return []
            
            strings = []
            try:
                if isinstance(obj, str):
                    strings.append(obj.lower())
                elif isinstance(obj, dict):
                    for v in obj.values():
                        strings.extend(extract_strings(v, depth + 1))
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        strings.extend(extract_strings(item, depth + 1))
                elif hasattr(obj, '__dict__'):
                    for v in obj.__dict__.values():
                        strings.extend(extract_strings(v, depth + 1))
                else:
                    strings.append(str(obj).lower())
            except:
                pass
            
            return strings
        
        all_text = extract_strings(self.target_object)
        return any(self._fuzzy_match(text, term) for text in all_text)


class SimpleLuceneFilter:
    """
    Simple Lucene object filter that focuses on the 80/20 rule.
    Uses luqum's native capabilities for parsing and tree manipulation.
    """
    
    def __init__(self):
        # Use luqum's built-in resolver for unknown operations
        self.operation_resolver = UnknownOperationResolver()
    
    def filter(self, objects: List[Any], query: str) -> List[Any]:
        """
        Filter objects using Lucene query syntax.
        
        Args:
            objects: List of Python objects to filter (dicts, dataclasses, etc.)
            query: Lucene query string
            
        Returns:
            List of objects that match the query
        """
        if not query or not query.strip():
            return objects
        
        try:
            # Parse query using luqum's parser
            ast = parser.parse(query)
            
            # Resolve unknown operations (implicit AND/OR)
            resolved_ast = self.operation_resolver(ast)
            
            # Filter objects
            results = []
            for obj in objects:
                transformer = ObjectFilterTransformer(obj)
                
                try:
                    if transformer.visit(resolved_ast):
                        results.append(obj)
                except Exception:
                    # Skip objects that cause errors during filtering
                    continue
            
            return results
            
        except Exception as e:
            # On any parsing or execution error, return empty list
            # This handles malformed queries gracefully
            return []
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate a query and return information about it.
        
        Returns:
            Dict with 'valid', 'error', and 'ast' keys
        """
        try:
            ast = parser.parse(query)
            resolved_ast = self.operation_resolver(ast)
            
            return {
                'valid': True,
                'error': None,
                'ast': str(resolved_ast)
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'ast': None
            }


# Example usage focusing on common use cases
if __name__ == "__main__":
    # Test data - common object structures
    test_data = [
        # Dictionaries (most common case)
        {
            "name": "web-server-01",
            "cpu": 85.5,
            "memory": 16,
            "status": "running",
            "tags": ["web", "production"],
            "config": {"ssl": True, "port": 443},
            "created": "2024-01-15"
        },
        {
            "name": "db-server-01", 
            "cpu": 45.2,
            "memory": 32,
            "status": "running",
            "tags": ["database", "production"],
            "config": {"ssl": False, "port": 5432},
            "created": "2024-01-10"
        },
        {
            "name": "cache-dev-01",
            "cpu": 25.0,
            "memory": 8,
            "status": "stopped",
            "tags": ["cache", "development"],
            "config": {"ssl": False, "port": 6379},
            "created": "2024-01-20"
        }
    ]
    
    # Dataclass example
    from dataclasses import dataclass
    
    @dataclass
    class Alert:
        severity: str
        message: str
        resolved: bool
        count: int
    
    alerts = [
        Alert("critical", "Database connection failed", False, 5),
        Alert("warning", "High CPU usage", True, 2),
        Alert("info", "Backup completed", True, 1)
    ]
    
    # Create filter
    lucene_filter = SimpleLuceneFilter()
    
    # Test common query patterns (80% use cases)
    test_queries = [
        # Basic field matching
        "status:running",
        "cpu:85.5",
        "tags:production",
        
        # Range queries
        "cpu:[40 TO 90]",
        "memory:[* TO 20]",
        
        # Boolean logic
        "status:running AND tags:production",
        "cpu:[60 TO *] OR memory:[30 TO *]",
        "NOT status:stopped",
        
        # Wildcards
        "name:*server*",
        "name:web*",
        
        # Nested fields
        "config.ssl:true",
        "config.port:443",
        
        # Array access
        "tags.0:web",
        
        # Full-text search
        "database",
        "production",
        
        # Fuzzy search
        "databas~",
        
        # Complex queries
        "(status:running AND tags:production) OR cpu:[80 TO *]",
        "config.ssl:true AND NOT status:stopped"
    ]
    
    print("=== Simple Lucene Object Filter Demo ===\n")
    
    print("Testing with dictionary objects:")
    for query in test_queries[:10]:  # Test first 10 queries
        try:
            results = lucene_filter.filter(test_data, query)
            names = [obj["name"] for obj in results]
            print(f"Query: {query}")
            print(f"Results: {names}")
            print()
        except Exception as e:
            print(f"Query: {query}")
            print(f"Error: {e}")
            print()
    
    print("\nTesting with dataclass objects:")
    dataclass_queries = ["severity:critical", "resolved:false", "count:[3 TO *]"]
    
    for query in dataclass_queries:
        try:
            results = lucene_filter.filter(alerts, query)
            descriptions = [f"{alert.severity}: {alert.message}" for alert in results]
            print(f"Query: {query}")
            print(f"Results: {descriptions}")
            print()
        except Exception as e:
            print(f"Query: {query}")
            print(f"Error: {e}")
            print()
    
    # Test query validation
    print("Query validation examples:")
    validation_tests = [
        "status:running",
        "invalid[[query",
        "(status:running AND",
        "field:[1 TO 10] OR other:value"
    ]
    
    for query in validation_tests:
        result = lucene_filter.validate_query(query)
        print(f"Query: {query}")
        print(f"Valid: {result['valid']}")
        if result['error']:
            print(f"Error: {result['error']}")
        print()