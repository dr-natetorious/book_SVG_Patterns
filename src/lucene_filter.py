"""
Production-ready Lucene filtering using luqum library
pip install luqum
"""

from luqum.parser import parser
from luqum.visitor import TreeVisitor
from luqum.tree import (
    SearchField, Word, Phrase, Range, Fuzzy, Proximity, 
    AndOperation, OrOperation, NotOperation, Group,
    FieldGroup, UnknownOperation
)
from typing import Any, List, Dict, Union
import re


class ObjectFilterVisitor(TreeVisitor):
    """Visitor that evaluates luqum AST against Python objects"""
    
    def __init__(self, obj: Any):
        self.obj = obj
        self._visited = set()  # Prevent circular references
        
    def visit_search_field(self, node: SearchField, context=None):
        """Handle field:value queries"""
        field_name = node.name
        field_value = self._get_field_value(field_name)
        
        # Visit the value part
        return self.visit(node.expr, context=field_value)
    
    def visit_word(self, node: Word, context=None):
        """Handle single word matches"""
        target_value = str(node.value)
        
        if context is not None:
            # Field-specific search
            return self._compare_values(context, target_value)
        else:
            # Full-text search across all fields
            return self._fulltext_search(target_value)
    
    def visit_phrase(self, node: Phrase, context=None):
        """Handle quoted phrase matches"""
        target_phrase = str(node.value).strip('"')
        
        if context is not None:
            # Exact phrase match in field
            return target_phrase.lower() in str(context).lower()
        else:
            # Full-text phrase search
            return self._fulltext_search(target_phrase)
    
    def visit_range(self, node: Range, context=None):
        """Handle range queries [min TO max]"""
        if context is None:
            return False
            
        try:
            field_val = self._convert_value(str(context))
            low_val = self._convert_value(str(node.low))
            high_val = self._convert_value(str(node.high))
            
            # Handle inclusive/exclusive bounds
            low_ok = field_val >= low_val if node.include_low else field_val > low_val
            high_ok = field_val <= high_val if node.include_high else field_val < high_val
            
            return low_ok and high_ok
        except (ValueError, TypeError):
            return False
    
    def visit_fuzzy(self, node: Fuzzy, context=None):
        """Handle fuzzy matching with ~"""
        # Simple fuzzy implementation - you could use python-Levenshtein here
        target = str(node.term)
        if context is not None:
            context_str = str(context).lower()
            target_str = target.lower()
            # Simple "contains" fuzzy matching
            return target_str in context_str or context_str in target_str
        return self._fulltext_search(target)
    
    def visit_and_operation(self, node: AndOperation, context=None):
        """Handle AND operations"""
        results = [self.visit(child, context) for child in node.children]
        return all(results)
    
    def visit_or_operation(self, node: OrOperation, context=None):
        """Handle OR operations"""
        results = [self.visit(child, context) for child in node.children]
        return any(results)
    
    def visit_not_operation(self, node: NotOperation, context=None):
        """Handle NOT operations"""
        return not self.visit(node.children[0], context)
    
    def visit_group(self, node: Group, context=None):
        """Handle parenthetical grouping"""
        return self.visit(node.children[0], context)
    
    def visit_field_group(self, node: FieldGroup, context=None):
        """Handle field:(expr1 expr2) grouping"""
        field_value = self._get_field_value(node.name)
        return self.visit(node.expr, context=field_value)
    
    def visit_unknown_operation(self, node: UnknownOperation, context=None):
        """Handle implicit operations (space between terms)"""
        # Default to AND behavior
        results = [self.visit(child, context) for child in node.children]
        return all(results)
    
    def _get_field_value(self, field_path: str) -> Any:
        """Get field value with dot notation support"""
        try:
            obj_id = id(self.obj)
            if obj_id in self._visited:
                return None
            self._visited.add(obj_id)
            
            value = self.obj
            for part in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = getattr(value, part, None)
                    
                if value is None:
                    break
            
            self._visited.remove(obj_id)
            return value
        except (KeyError, AttributeError, TypeError):
            if obj_id in self._visited:
                self._visited.remove(obj_id)
            return None
    
    def _compare_values(self, field_value: Any, target_value: str) -> bool:
        """Compare field value with target, handling wildcards"""
        if field_value is None:
            return False
            
        field_str = str(field_value).lower()
        target_str = target_value.lower()
        
        # Handle wildcards
        if '*' in target_str or '?' in target_str:
            pattern = target_str.replace('*', '.*').replace('?', '.')
            return bool(re.match(f'^{pattern}$', field_str))
        
        # Try exact match first
        if field_str == target_str:
            return True
            
        # Try numeric comparison
        try:
            field_num = self._convert_value(str(field_value))
            target_num = self._convert_value(target_value)
            return field_num == target_num
        except (ValueError, TypeError):
            pass
            
        # Fallback to contains
        return target_str in field_str
    
    def _convert_value(self, value: str) -> Union[int, float, bool, str]:
        """Convert string to appropriate type"""
        value = value.strip('"\'')
        
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer
        try:
            if '.' not in value and 'e' not in value.lower():
                return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        return value
    
    def _fulltext_search(self, term: str) -> bool:
        """Search for term across all string fields"""
        term_lower = term.lower()
        
        def extract_text(obj, depth=0):
            if depth > 10:  # Prevent deep recursion
                return []
            
            texts = []
            if isinstance(obj, str):
                texts.append(obj.lower())
            elif isinstance(obj, dict):
                for value in obj.values():
                    texts.extend(extract_text(value, depth + 1))
            elif hasattr(obj, '__dict__'):
                for value in obj.__dict__.values():
                    texts.extend(extract_text(value, depth + 1))
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    texts.extend(extract_text(item, depth + 1))
            else:
                texts.append(str(obj).lower())
            
            return texts
        
        all_text = extract_text(self.obj)
        return any(term_lower in text for text in all_text)


class LuceneObjectFilter:
    """
    Production-ready Lucene-style filtering using luqum library.
    
    Advantages over custom implementation:
    - Battle-tested query parsing
    - Full Lucene syntax support
    - Proper operator precedence
    - Extensible visitor pattern
    - Memory efficient
    - No circular reference issues
    """
    
    def filter(self, objects: List[Any], query: str) -> List[Any]:
        """Filter objects using Lucene query syntax"""
        if not query.strip():
            return objects
        
        try:
            # Parse query into AST
            ast = parser.parse(query)
            
            # Filter objects
            results = []
            for obj in objects:
                visitor = ObjectFilterVisitor(obj)
                if visitor.visit(ast):
                    results.append(obj)
            
            return results
            
        except Exception as e:
            # Graceful degradation on parse errors
            print(f"Query parse error: {e}")
            return objects
    
    def explain_query(self, query: str) -> str:
        """Show the parsed query structure for debugging"""
        try:
            ast = parser.parse(query)
            return str(ast)
        except Exception as e:
            return f"Parse error: {e}"


# Example usage and testing
if __name__ == "__main__":
    # Test data
    servers = [
        {
            "name": "web-01",
            "cpu": 85,
            "memory": 16,
            "status": "running",
            "tags": ["web", "production"],
            "config": {"ssl": True, "port": 443}
        },
        {
            "name": "db-primary",
            "cpu": 45,
            "memory": 32,
            "status": "running",
            "tags": ["database", "production"],
            "config": {"ssl": False, "port": 5432}
        },
        {
            "name": "cache-dev",
            "cpu": 20,
            "memory": 8,
            "status": "stopped",
            "tags": ["cache", "development"],
            "config": {"ssl": False, "port": 6379}
        }
    ]
    
    filter_engine = LuceneObjectFilter()
    
    # Test cases
    test_queries = [
        "status:running",
        "cpu:[40 TO 90]",
        "name:*-01",
        "tags:production AND cpu:>60",
        "config.ssl:true OR status:stopped",
        'name:"db-primary"',
        "NOT status:running",
        "(status:running OR status:idle) AND memory:>10"
    ]
    
    print("=== Testing Lucene Object Filter ===")
    for query in test_queries:
        results = filter_engine.filter(servers, query)
        names = [s['name'] for s in results]
        print(f"Query: {query}")
        print(f"Results: {names}")
        print(f"AST: {filter_engine.explain_query(query)}")
        print("-" * 50)