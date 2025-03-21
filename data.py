import json
import os
import re
import numpy as np
from collections import defaultdict
from typing import Dict

from config import (
    RESULT_DIR, 
    color_mapping, 
    path_dict_cache, 
    plot_data_cache
)

# Caching functions (moved up to fix linter errors)
def make_hashable(obj):
    """Convert mutable objects to immutable for hashing."""
    if isinstance(obj, list):
        # Try to sort the list if elements are comparable
        try:
            # Convert each element to hashable and then sort
            return tuple(sorted(make_hashable(item) for item in obj))
        except TypeError:
            # If elements aren't comparable (like dicts), just convert to tuple without sorting
            return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        # Sort dictionary items by key for consistent hashing
        return tuple(sorted((key, make_hashable(value)) for key, value in obj.items()))
    elif isinstance(obj, set):
        # Convert set to sorted frozenset
        return frozenset(sorted(make_hashable(item) for item in obj))
    else:
        return obj

def custom_cached(cache):
    """Decorator that ensures all arguments are hashable before caching."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Convert args to hashable types
            hashable_args = tuple(make_hashable(arg) for arg in args)
            
            # Convert kwargs to hashable types
            hashable_kwargs = {key: make_hashable(value) for key, value in kwargs.items()}
            hashable_kwargs_tuple = tuple(sorted(hashable_kwargs.items()))
            
            # Create a cache key from the hashable args and kwargs
            key = (func.__name__, hashable_args, hashable_kwargs_tuple)
            
            # Debug logging for time_bucket
            if func.__name__ == 'get_category_time_series' and 'time_bucket' in kwargs:
                print(f"Cache key includes time_bucket: {kwargs['time_bucket']}")
                print(f"Using cache key: {key}")
            
            # Check if result is in cache
            if key in cache:
                if func.__name__ == 'get_category_time_series':
                    print(f"Cache hit for {func.__name__} with time_bucket={kwargs.get('time_bucket', 'default')}")
                return cache[key]
            
            # Call the function with original arguments
            result = func(*args, **kwargs)
            
            # Store result in cache
            cache[key] = result
            
            if func.__name__ == 'get_category_time_series':
                print(f"Cache miss for {func.__name__}, stored result with time_bucket={kwargs.get('time_bucket', 'default')}")
            
            return result
        return wrapper
    return decorator

# Load data
use_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'use_path_to_sents_dict.json')))
attr_perf_path_to_ids_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_perf_path_to_ids_dict.json')))
attr_perf_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_perf_path_to_sents_dict.json')))
perf_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'perf_path_to_sents_dict.json')))
attr_path_to_ids_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_path_to_ids_dict.json')))
attr_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_path_to_sents_dict.json')))

# Map type to raw dictionaries
raw_dict_map = {
    'use_sents': use_path_to_sents_dict,
    'perf_sents': perf_path_to_sents_dict,
    'attr_sents': attr_path_to_sents_dict,
    'attr_perf_sents': attr_perf_path_to_sents_dict,
    'attr_ids': attr_path_to_ids_dict,
    'attr_perf_ids': attr_perf_path_to_ids_dict
}

def merge_values(val1, val2):
    """Merge two values together, handling different types (lists, dicts, etc)."""
    if val1 is None or len(val1) == 0:
        return val2
    if val2 is None or len(val2) == 0:
        return val1
    if isinstance(val1, list) or isinstance(val2, list):
        if isinstance(val1, list) and isinstance(val2, list):
            return list(set(val1+val2))
        elif hasattr(val1, 'keys'):
            return merge_values(val1, {'_': val2})
        else:
            return merge_values({'_': val1}, val2)
    elif hasattr(val1, 'keys') and hasattr(val2, 'keys'):
        new_dict = {}
        for key in set(list(val1.keys())+list(val2.keys())):
            new_dict[key] = merge_values(val1.get(key, None), val2.get(key, None))
        return new_dict
    else:
        raise ValueError(f"Invalid value type: {type(val1)} of {val1} and {type(val2)} of {val2}")

def filter_dict(path_to_sents_dict, category):
    """Filter a dictionary based on a category path."""
    if category == 'all':
        key_part_num = 1
    else:
        key_part_num = len(category.split('|'))+1
    filtered_dict = defaultdict(dict)
    for path, sents_dict in path_to_sents_dict.items():
        if category == 'all' or path.startswith(category):
            key = '|'.join(path.split('|')[:key_part_num])
            filtered_dict[key] = merge_values(filtered_dict[key], sents_dict)
    return filtered_dict


def extract_review_info(review):
    """
    Extract review text, highlight information, and date in a single pass.
    
    Args:
        review: Full review string in the format "highlight|||review_text"
        
    Returns:
        Tuple of (review_text, highlight_detail, highlight_reason, date)
        Where:
          - review_text: The main review text
          - highlight_detail: The extracted highlight detail
          - highlight_reason: The extracted highlight reason or None
          - date: The extracted date in format YYYY-MM-DD or None
    """
    # Extract highlight information
    highlight, review_text = review.split('|||')
    if '<<<' in highlight:
        highlight_detail, highlight_reason = highlight.split('<<<')
    else:
        highlight_detail, highlight_reason = highlight, None
    
    # Extract date
    date_match = re.search(r'\[(\d{4}-\d{2}-\d{2})\]', review_text)
    date = date_match.group(1) if date_match else None
    
    return review_text, highlight_detail, highlight_reason, date

def reviews_to_htmls(review_to_highlight_dict, detail_axis='x', display_reason=True):
    """Convert reviews to HTML format with highlighting."""
    review_htmls = []
    pos_count = 0
    for review_text, highlights in review_to_highlight_dict.items():
        highlight_htmls = []
        pos_count_list = []
        detail_to_sent = {}
        reason_to_sent = {}
        for sent, highlight_detail, highlight_reason in highlights:
            if sent == '+':
                pos_count_list.append(1)
            elif sent == '-':
                pos_count_list.append(0)
            else:
                # print(f"Warning: {sent} is not a valid sentiment for {highlight_detail}|||{review_text}")
                pos_count_list.append(0.5)
                sent = '?'
            detail_to_sent[highlight_detail] = sent
            if highlight_reason and display_reason:
                reason_to_sent[highlight_reason] = sent
        highlight_htmls.extend([f'<span style="background-color: {color_mapping[sent]}; border: 4px solid {color_mapping[detail_axis]}; color: black; padding: 2px;">{highlight_detail}</span>' for highlight_detail in detail_to_sent.keys()])
        if display_reason:
            highlight_htmls.extend([f'<span style="background-color: {color_mapping[sent]}; border: 4px solid {color_mapping["y"]}; color: black; padding: 2px;">{highlight_reason}</span>' for highlight_reason in reason_to_sent.keys()])
        review_htmls.append(f"{' '.join(highlight_htmls)} {review_text}")
        pos_count += sum(pos_count_list)/len(pos_count_list)
    return review_htmls, pos_count

def create_correlation_matrix(x_path_to_sents_dict, y_path_to_ids_dict, y_path_to_sents_dict):
    """Create correlation matrix between x and y categories."""
    matrix = np.zeros((len(y_path_to_ids_dict), len(x_path_to_sents_dict)))
    sentiment_matrix = np.zeros((len(y_path_to_ids_dict), len(x_path_to_sents_dict)))
    review_matrix = np.empty((len(y_path_to_ids_dict), len(x_path_to_sents_dict)), dtype=object)
    
    # Initialize each cell with its own empty list
    for i in range(review_matrix.shape[0]):
        for j in range(review_matrix.shape[1]):
            review_matrix[i, j] = []
    
    for i, ((path1, y_ids), y_sents_dict) in enumerate(zip(y_path_to_ids_dict.items(), y_path_to_sents_dict.values())):
        for j, (path2, x_sents_dict) in enumerate(x_path_to_sents_dict.items()):
            x_sent_ids = set()
            review_to_highlight_dict = defaultdict(list)
            
            # First collect all reviews and sentiments for this x category
            for sent, reason_rids in x_sents_dict.items():
                x_sent_ids.update(reason_rids.keys())
                for rid, reviews in reason_rids.items():
                    if rid in y_ids:  # Co-mention case
                        # Check if reviews is None before iterating
                        if reviews is None:
                            continue
                        for review in reviews:
                            # Check if this is a pre-processed review or raw string
                            if isinstance(review, tuple) and len(review) == 5:
                                # Already processed: (review_text, highlight, reason, date, raw_review)
                                review_text, highlight_detail, highlight_reason, _, _ = review
                            else:
                                # Raw string, extract info
                                review_text, highlight_detail, highlight_reason, _ = extract_review_info(review)
                            
                            review_to_highlight_dict[review_text].append((sent, highlight_detail, highlight_reason))
            review_with_highlight, pos_count = reviews_to_htmls(review_to_highlight_dict, detail_axis='x', display_reason=True)
            review_matrix[i, j] = review_with_highlight
            sentiment_matrix[i, j] = pos_count
            # Update counts and sentiment
            matrix[i, j] = len(review_with_highlight)
            if matrix[i, j] > 0:
                sentiment_matrix[i, j] = pos_count / matrix[i, j]
    
    return matrix, sentiment_matrix, review_matrix

@custom_cached(cache=path_dict_cache)
def get_cached_dict(plot_type: str, category: str, search_query: str = '', start_date: str = None, end_date: str = None, return_both: bool = False) -> Dict:
    """
    Get filtered dictionary from cache. Can return both sents and ids dicts for attr/attr_perf.
    """
    raw_dict = raw_dict_map[plot_type]
    
    # Filter by category
    filtered_dict = filter_dict(raw_dict, category)
    
    # Only apply search query filter to sents dictionaries
    if (search_query and search_query.strip() and not plot_type.endswith('_ids')) or start_date or end_date:
        filtered_dict = filter_dict_by_query(filtered_dict, search_query, start_date, end_date)
        
    # If we need both sents and ids dicts
    if return_both:
        if plot_type == 'attr_perf_sents':
            ids_dict = filter_dict(attr_perf_path_to_ids_dict, category)
            # Filter ids_dict to match paths in filtered sents dict
            ids_dict = {path: ids for path, ids in ids_dict.items() 
                       if path in filtered_dict}
        elif plot_type == 'attr_sents':
            ids_dict = filter_dict(attr_path_to_ids_dict, category)
            # Filter ids_dict to match paths in filtered sents dict
            ids_dict = {path: ids for path, ids in ids_dict.items() 
                       if path in filtered_dict}
        else:
            raise ValueError(f"Cannot return both dicts for plot type: {plot_type}")
            
        return filtered_dict, ids_dict
        
    return filtered_dict

@custom_cached(cache=plot_data_cache)
def get_plot_data(plot_type, x_category='all', y_category='all', top_n_x=2, top_n_y=2, language='en', search_query='', start_date=None, end_date=None):
    """Get data for plotting the correlation matrix."""
    # Convert slider values to integers
    top_n_x = int(top_n_x)
    top_n_y = int(top_n_y)
    
    if plot_type == 'use_attr_perf':
        x_path_to_sents_dict = get_cached_dict('use_sents', x_category, search_query, start_date, end_date)
        y_path_to_sents_dict, y_path_to_ids_dict = get_cached_dict(
            'attr_perf_sents', y_category, search_query, start_date, end_date, return_both=True
        )
        title_key = 'use_attr_perf_title'
    else:  # perf_attr
        x_path_to_sents_dict = get_cached_dict('perf_sents', x_category, search_query, start_date, end_date)
        y_path_to_sents_dict, y_path_to_ids_dict = get_cached_dict(
            'attr_sents', y_category, search_query, start_date, end_date, return_both=True
        )
        title_key = 'perf_attr_title'

    matrix, sentiment_matrix, review_matrix = create_correlation_matrix(
        x_path_to_sents_dict, 
        y_path_to_ids_dict,
        y_path_to_sents_dict
    )
    
    def process_axis(matrix: np.ndarray, axis: int, top_n: int, paths: list) -> tuple:
        """Process one axis (x or y) to get indices, paths and percentages."""
        sums = np.sum(matrix, axis=1-axis)  # sum over opposite axis
        non_zero = np.where(sums > 0)[0]
        if len(non_zero) == 0:
            # If there are no non-zero values, show the first top_n features
            top_indices = np.arange(min(top_n, len(sums)))
        else:
            top_indices = non_zero[np.argsort(sums[non_zero])[-min(top_n, len(non_zero)):]]
        top_paths = [paths[i] for i in top_indices]
        total_mentions = np.sum(matrix)
        percentages = np.sum(matrix[top_indices] if axis == 0 else matrix[:, top_indices], axis=1-axis) / total_mentions * 100 if total_mentions > 0 else np.zeros(len(top_indices))
        return top_indices, top_paths, percentages
    
    # Process x and y axes
    x_paths = list(x_path_to_sents_dict.keys())
    y_paths = list(y_path_to_ids_dict.keys())
    top_x_indices, top_x_paths, x_percentages = process_axis(matrix, 1, top_n_x, x_paths)
    top_y_indices, top_y_paths, y_percentages = process_axis(matrix, 0, top_n_y, y_paths)
    
    # Extract relevant submatrices
    matrix = matrix[top_y_indices, :][:, top_x_indices]
    sentiment_matrix = sentiment_matrix[top_y_indices, :][:, top_x_indices]
    review_matrix = review_matrix[top_y_indices, :][:, top_x_indices]
    
    return matrix, sentiment_matrix, review_matrix, top_x_paths, x_percentages, top_y_paths, y_percentages, title_key

def filter_dict_by_query(path_to_sents_dict: Dict, search_query: str, start_date: str = None, end_date: str = None) -> Dict:
    """
    Filter path_to_sents_dict by removing reviews that don't match the query or date range
    
    Extracts all review information (text, highlight, reason, date) during filtering to
    avoid re-parsing reviews later.
    """
    if (not search_query or not search_query.strip()) and start_date is None and end_date is None:
        return path_to_sents_dict
    
    # Function to extract and filter review information
    def extract_and_filter_review(review, query_filter=None):
        """
        Extract info from a review and apply filters
        
        Args:
            review: The review to process
            query_filter: Optional function to filter by query text
            
        Returns:
            Processed review tuple or None if filtered out
        """
        # Extract all review information
        review_text, highlight, reason, date = extract_review_info(review)
        
        # Apply date filter
        if start_date is not None or end_date is not None:
            if date is None or (start_date and date < start_date) or (end_date and date > end_date):
                return None
        
        # Apply search query filter if provided
        if query_filter and not query_filter(review_text.lower()):
            return None
        
        # Return the processed review with all extracted info
        return (review_text, highlight, reason, date, review)
    
    # Function to process a list of reviews
    def process_reviews(reviews, query_filter=None):
        """Process reviews: extract info and apply filters"""
        if reviews is None:
            return []
        
        processed_reviews = []
        for review in reviews:
            processed_review = extract_and_filter_review(review, query_filter)
            if processed_review:  # Only keep reviews that pass filters
                processed_reviews.append(processed_review)
        
        return processed_reviews
    
    # Create query filter function if search query is provided
    query_filter = None
    if search_query and search_query.strip():
        try:
            # Convert operators and clean up query
            query = search_query.lower().strip()
            query = re.sub(r'\s*&\s*', ' and ', query)
            query = re.sub(r'\s*\|\s*', ' or ', query)
            
            # Parse terms in quotes as single terms
            quoted_terms = re.findall(r'"([^"]+)"', query)
            for term in quoted_terms:
                if ' ' in term:  # Only replace if it has spaces
                    query = query.replace(f'"{term}"', f'"{term.replace(" ", "_")}"')
            
            # Convert terms to regex patterns
            terms = []
            current_group = []
            
            for term in re.findall(r'"[^"]+"|\w+|[()&|]', query):
                if term in ('and', 'or', '(', ')'):
                    # If we've collected terms and hit an operator, join them with OR
                    if current_group and term not in ('(', ')'):
                        if len(current_group) > 1:
                            terms.append(f"({' or '.join(current_group)})")
                        else:
                            terms.append(current_group[0])
                        current_group = []
                        
                    terms.append(term)
                else:
                    # Handle quoted terms
                    if term.startswith('"') and term.endswith('"'):
                        clean_term = term[1:-1].replace('_', ' ')  # Convert back underscores to spaces
                        current_group.append(f"'{clean_term}' in review_text")
                    else:
                        current_group.append(f"'{term}' in review_text")
            
            # Handle any remaining terms
            if current_group:
                if len(current_group) > 1:
                    terms.append(f"({' or '.join(current_group)})")
                else:
                    terms.append(current_group[0])
            
            # Create the filter expression
            filter_expr = ' '.join(terms)
            
            # Handle edge cases with parentheses
            filter_expr = re.sub(r'\(\s+', '(', filter_expr)
            filter_expr = re.sub(r'\s+\)', ')', filter_expr)
            
            # Force spacing around operators for proper evaluation
            filter_expr = re.sub(r'(\))\s*(\()', r'\1 and \2', filter_expr)
            
            # Create a compiled filter function to avoid re-evaluation
            filter_code = compile(f"lambda review_text: {filter_expr}", "<string>", "eval")
            query_filter = eval(filter_code, {"__builtins__": {}})
            
        except Exception as e:
            print(f"Search query error: {str(e)}")
            print(f"Filter expression was: {filter_expr if 'filter_expr' in locals() else 'not created'}")
            query_filter = None  # Fall back to just date filtering
    
    # Now apply the filter to the dictionary structure
    filtered_dict = {}
    for path, sents_dict in path_to_sents_dict.items():
        # Handle case where value is a list (ids_dict case)
        if isinstance(sents_dict, list):
            # For ids_dict, we keep the path if any associated reviews match
            # We'll need to look up the reviews in the corresponding sents_dict
            filtered_dict[path] = sents_dict
            continue
            
        # Handle case where value is a dict (sents_dict case)
        filtered_sents = {}
        for sent, val in sents_dict.items():
            if isinstance(val, dict):
                rid_reviews_dict = val
                filtered_rids = {}
                for rid, reviews in rid_reviews_dict.items():
                    processed_reviews = process_reviews(reviews, query_filter)
                    if processed_reviews:  # Only keep if there are matching reviews
                        filtered_rids[rid] = processed_reviews
                if filtered_rids:  # Only keep sentiment if there are matching reviews
                    filtered_sents[sent] = filtered_rids
            elif isinstance(val, list):
                reviews = val
                processed_reviews = process_reviews(reviews, query_filter)
                if processed_reviews:
                    filtered_sents[sent] = processed_reviews

        if filtered_sents:  # Only keep path if there are matching reviews
            filtered_dict[path] = filtered_sents
            
    return filtered_dict

def normalize_search_query(query: str) -> str:
    """Normalize search query for display by quoting terms and using readable operators"""
    if not query or not query.strip():
        return ""
        
    query = query.strip()
    
    # First handle parentheses spacing
    query = re.sub(r'\(\s*', '(', query)
    query = re.sub(r'\s*\)', ')', query)
    
    # Preserve existing quoted terms
    quoted_sections = {}
    quote_pattern = re.compile(r'"([^"]+)"')
    quoted_matches = quote_pattern.findall(query)
    
    for i, match in enumerate(quoted_matches):
        placeholder = f"QUOTED_{i}"
        quoted_sections[placeholder] = match
        query = query.replace(f'"{match}"', placeholder)
    
    # Split into tokens while preserving operators and parentheses
    tokens = re.findall(r'\(|\)|QUOTED_\d+|\w+|&|\|', query)
    
    # Process tokens
    normalized = []
    for token in tokens:
        if token in ('&', '|'):
            # Replace operators
            normalized.append('and' if token == '&' else 'or')
        elif token in ('(', ')'):
            # Keep parentheses as-is
            normalized.append(token)
        elif token.startswith('QUOTED_'):
            # Restore quoted term
            normalized.append(f'"{quoted_sections[token]}"')
        elif ' ' in token:
            # Quote multi-word terms
            normalized.append(f'"{token}"')
        else:
            # Quote single terms
            normalized.append(f'"{token}"')
    
    # Join with proper spacing
    return ' '.join(normalized)

# Use our custom caching decorator instead of the built-in @cached
@custom_cached(cache=plot_data_cache)
def get_bar_chart_data(categories, zoom_category=None, language='en', search_query='', start_date=None, end_date=None):
    """
    Get data for bar chart visualization.
    
    Args:
        categories: List of categories to include (usage, attribute, performance)
        zoom_category: Optional category to zoom in on (e.g., "usage|category1")
        language: Language for translations
        search_query: Optional search query to filter results
        start_date: Optional start date to filter reviews
        end_date: Optional end date to filter reviews
    
    Returns:
        Tuple containing (display_categories, original_categories, counts, sentiment_ratios, review_data, colors, title_key)
        Where display_categories are formatted for display and original_categories preserve the raw category values
    """
    # Use a dictionary for more efficient lookup
    result_dict = {}
    
    # Treat empty string as None
    if zoom_category == '' or zoom_category == 'all':
        zoom_category = None
    
    # Map category types to their dictionaries and colors
    category_mapping = {
        'usage': {
            'dict_type': 'use_sents',
            'prefix': 'U: ',
            'color': '#4285F4'  # Blue
        },
        'attribute': {
            'dict_type': 'attr_sents', 
            'prefix': 'A: ',
            'color': '#34A853'  # Green
        },
        'performance': {
            'dict_type': 'perf_sents',
            'prefix': 'P: ',
            'color': '#FBBC05'  # Yellow
        }
    }
    
    # Helper function to process a single review and add it to the results
    def process_review(review, sent, total_reviews, positive_reviews, review_highlights):
        """
        Process a single review and update the tracking collections
        
        Args:
            review: The review to process
            sent: The sentiment value
            total_reviews: Set to track unique reviews
            positive_reviews: Set to track positive reviews
            review_highlights: List to collect review highlights
        """
        # Check if this is a pre-processed review or raw string
        if isinstance(review, tuple) and len(review) == 5:
            # Already processed: (review_text, highlight, reason, date, raw_review)
            review_text, highlight, reason, _, raw_review = review
        else:
            # Raw string, extract info
            review_text, highlight, reason, _ = extract_review_info(review)
            raw_review = review
            
        total_reviews.add(review_text)
        
        # Check if positive review
        if '+' in highlight:
            positive_reviews.add(review_text)
            
        # Add to highlight list with context
        review_to_highlight_dict = defaultdict(list)
        review_to_highlight_dict[review_text].append((sent, highlight, reason))
        review_with_highlight, review_pos_count = reviews_to_htmls(review_to_highlight_dict, detail_axis='x', display_reason=True)
        review_highlights.extend(review_with_highlight)
        
        # Add sentiment value to positive_reviews if positive (for proper sentiment calculation)
        if sent == '+':
            positive_reviews.add(review_text)
    # TODO: fix the review_highlights by merging the same reviews from different categories
    # Process each selected category type
    for category_type in categories:
        if category_type not in category_mapping:
            continue
            
        mapping = category_mapping[category_type]
        dict_type = mapping['dict_type']
        prefix = mapping['prefix']
        color = mapping['color']
        
        # Get the dictionary for this category type
        if zoom_category:
            # If zooming in on a category, only include subcategories of that category
            if zoom_category.startswith(mapping['prefix']):
                category_path = zoom_category[len(mapping['prefix']):]
                category_dict = get_cached_dict(dict_type, category_path, search_query, start_date, end_date)
            else:
                # Skip if zooming into a different category type
                continue
        else:
            # Otherwise get all categories
            category_dict = get_cached_dict(dict_type, 'all', search_query, start_date, end_date)
        
        # Count mentions for each category
        for category_path, sents_dict in category_dict.items():
            # Store the original category path for reference
            original_category = category_path
            
            # Get the display name for the category
            if zoom_category:
                # When zoomed in, preserve the full hierarchy
                # For example: if category_path is "parent|child", display as "parent|child"
                if '|' in category_path:
                    display_name = prefix + category_path
                else:
                    # If it's a leaf category with no children, just show the category
                    display_name = prefix + category_path
            else:
                # For top-level view, we want to distinguish subcategories
                if '|' in category_path:
                    # Show full hierarchy for subcategories
                    parts = category_path.split('|')
                    # Make sure we show the parent|child relationship
                    display_name = prefix + parts[0] + '|' + parts[1]
                else:
                    # Just show the top-level category
                    display_name = prefix + category_path
                
            # Count total reviews and positive reviews
            total_reviews = set()
            positive_reviews = set()
            review_highlights = []
            
            # Process all reviews for this category
            for sent, reason_rids in sents_dict.items():
                # Check if reason_rids is a dictionary or a list
                if isinstance(reason_rids, dict):
                    # It's a dictionary with rid -> reviews mapping
                    for rid, reviews in reason_rids.items():
                        if reviews is None:
                            continue
                        for review in reviews:
                            process_review(review, sent, total_reviews, positive_reviews, review_highlights)
                elif isinstance(reason_rids, list):
                    # It's a list of reviews directly
                    if reason_rids is None:
                        continue
                    for review in reason_rids:
                        process_review(review, sent, total_reviews, positive_reviews, review_highlights)
            
            # Only include categories with reviews
            if total_reviews:
                count = len(total_reviews)
                sentiment_ratio = len(positive_reviews) / count if count > 0 else 0
                
                # Check if category already exists in result dictionary (for aggregation)
                if display_name in result_dict:
                    # If exists, add to count and update sentiment ratio
                    existing_count = result_dict[display_name]['count']
                    existing_pos_count = result_dict[display_name]['sentiment_ratio'] * existing_count
                    new_total_count = existing_count + count
                    new_pos_count = existing_pos_count + len(positive_reviews)
                    
                    result_dict[display_name]['count'] = new_total_count
                    result_dict[display_name]['sentiment_ratio'] = new_pos_count / new_total_count
                    result_dict[display_name]['reviews'].extend(review_highlights)
                else:
                    # If not exists, add as new category
                    result_dict[display_name] = {
                        'original_category': prefix + original_category,
                        'count': count,
                        'sentiment_ratio': sentiment_ratio,
                        'reviews': review_highlights,
                        'color': color
                    }
    
    # Convert dictionary to sorted lists
    items = list(result_dict.items())
    # Sort by count in descending order
    items.sort(key=lambda x: x[1]['count'], reverse=True)
    
    # Create the result lists
    display_categories = []
    original_categories = []
    counts = []
    sentiment_ratios = []
    review_data = []
    colors = []
    
    for display_name, item in items:
        display_categories.append(display_name)
        original_categories.append(item['original_category'])
        counts.append(item['count'])
        sentiment_ratios.append(item['sentiment_ratio'])
        review_data.append(item['reviews'])
        colors.append(item['color'])
    
    return display_categories, original_categories, counts, sentiment_ratios, review_data, colors, 'bar_chart_title'

def get_review_date_range():
    """Find the earliest and latest dates from all review dictionaries."""
    min_date = None
    max_date = None

    # Process all review dictionaries to find date range
    for dict_type in ['use_sents', 'attr_perf_sents']:
        path_to_sents_dict = raw_dict_map[dict_type]
        
        for path, sents_dict in path_to_sents_dict.items():
            for sent, val in sents_dict.items():
                if isinstance(val, dict):
                    # It's a dictionary with rid -> reviews mapping
                    for rid, reviews in val.items():
                        if reviews is None:
                            continue
                        for review in reviews:
                            _, _, _, date = extract_review_info(review)
                            if date:
                                if min_date is None or date < min_date:
                                    min_date = date
                                if max_date is None or date > max_date:
                                    max_date = date
                elif isinstance(val, list):
                    # It's a list of reviews directly
                    if val is None:
                        continue
                    for review in val:
                        _, _, _, date = extract_review_info(review)
                        if date:
                            if min_date is None or date < min_date:
                                min_date = date
                            if max_date is None or date > max_date:
                                max_date = date
    
    return min_date, max_date

@custom_cached(cache=plot_data_cache)
def get_category_time_series(categories, display_categories, original_categories, zoom_category=None, 
                             language='en', search_query='', start_date=None, end_date=None, 
                             time_bucket='month'):
    """
    Get time series data for categories to display trend line chart.
    
    Args:
        categories: List of category types to include (usage, attribute, performance)
        display_categories: List of category names displayed in the bar chart
        original_categories: List of original category values (with prefixes)
        zoom_category: Optional category to zoom in on
        language: Language for translations
        search_query: Optional search query to filter results
        start_date: Optional start date to filter reviews
        end_date: Optional end date to filter reviews
        time_bucket: How to group time data ('day', 'week', 'month', 'year', '3month', '6month')
    
    Returns:
        Dictionary with dates and count/sentiment data for each category
    """
    from datetime import datetime
    import pandas as pd
    
    print(f"get_category_time_series called with time_bucket={time_bucket}")
    
    # Default to month if invalid bucket size provided
    valid_buckets = ['day', 'week', 'month', '3month', '6month', 'year']
    if time_bucket not in valid_buckets:
        print(f"Invalid time_bucket '{time_bucket}', defaulting to 'month'")
        time_bucket = 'month'
    else:
        # Force the time_bucket to be a string to avoid any potential typing issues with cache keys
        time_bucket = str(time_bucket)
    
    # Clear the cache when time_bucket changes to force a refresh
    # This is a workaround for any potential caching issues
    if time_bucket != 'month':  # Only clear for non-default values to avoid clearing on initial load
        plot_data_cache.clear()
    
    # Treat empty string as None
    if zoom_category == '' or zoom_category == 'all':
        zoom_category = None
    
    # Create mapping of display categories to find their data
    display_to_original = {display: original for display, original in zip(display_categories, original_categories)}
    
    # Map category types to their dictionaries
    category_mapping = {
        'usage': {'dict_type': 'use_sents', 'prefix': 'U: '},
        'attribute': {'dict_type': 'attr_sents', 'prefix': 'A: '},
        'performance': {'dict_type': 'perf_sents', 'prefix': 'P: '}
    }
    
    # Initialize time series data
    time_series_data = {
        'dates': [],
        'category_data': {cat: {'counts': [], 'sentiment': []} for cat in display_categories}
    }
    
    # Track all reviews by date and category
    date_category_reviews = {}
    date_format = '%Y-%m-%d'
    
    # Process each selected category type
    for category_type in categories:
        if category_type not in category_mapping:
            continue
            
        mapping = category_mapping[category_type]
        dict_type = mapping['dict_type']
        prefix = mapping['prefix']
        
        # Get the dictionary for this category type
        if zoom_category:
            # If zooming in on a category, only include subcategories of that category
            if zoom_category.startswith(mapping['prefix']):
                category_path = zoom_category[len(mapping['prefix']):]
                category_dict = get_cached_dict(dict_type, category_path, search_query, start_date, end_date)
            else:
                # Skip if zooming into a different category type
                continue
        else:
            # Otherwise get all categories
            category_dict = get_cached_dict(dict_type, 'all', search_query, start_date, end_date)
        
        # Process each category path
        for category_path, sents_dict in category_dict.items():
            # Get display name similar to bar chart logic
            if zoom_category:
                if '|' in category_path:
                    display_name = prefix + category_path
                else:
                    display_name = prefix + category_path
            else:
                if '|' in category_path:
                    parts = category_path.split('|')
                    display_name = prefix + parts[0] + '|' + parts[1]
                else:
                    display_name = prefix + category_path
                    
            # Skip if this category is not in our display categories list
            if display_name not in display_categories:
                continue
                
            # Process reviews for this category to collect date information
            for sent, reason_rids in sents_dict.items():
                if isinstance(reason_rids, dict):
                    # Dictionary with rid -> reviews mapping
                    for rid, reviews in reason_rids.items():
                        if reviews is None:
                            continue
                        for review in reviews:
                            # Check if this is a pre-processed review or raw string
                            if isinstance(review, tuple) and len(review) == 5:
                                # Already processed: (review_text, highlight, reason, date, raw_review)
                                _, highlight, _, date, _ = review
                            else:
                                # Raw string, extract info
                                _, highlight, _, date = extract_review_info(review)
                                
                            if date:
                                # Initialize date entry if needed
                                if date not in date_category_reviews:
                                    date_category_reviews[date] = {}
                                    
                                # Initialize category for this date if needed
                                if display_name not in date_category_reviews[date]:
                                    date_category_reviews[date][display_name] = {
                                        'count': 0,
                                        'positive': 0
                                    }
                                    
                                # Count review and track sentiment
                                date_category_reviews[date][display_name]['count'] += 1
                                if '+' == sent:
                                    date_category_reviews[date][display_name]['positive'] += 1
                elif isinstance(reason_rids, list):
                    # List of reviews directly
                    if reason_rids is None:
                        continue
                    for review in reason_rids:
                            # Check if this is a pre-processed review or raw string
                            if isinstance(review, tuple) and len(review) == 5:
                                # Already processed: (review_text, highlight, reason, date, raw_review)
                                _, highlight, _, date, _ = review
                            else:
                                # Raw string, extract info
                                _, highlight, _, date = extract_review_info(review)
                                
                            if date:
                                # Initialize date entry if needed
                                if date not in date_category_reviews:
                                    date_category_reviews[date] = {}
                                    
                                # Initialize category for this date if needed
                                if display_name not in date_category_reviews[date]:
                                    date_category_reviews[date][display_name] = {
                                        'count': 0,
                                        'positive': 0
                                    }
                                    
                                # Count review and track sentiment
                                date_category_reviews[date][display_name]['count'] += 1
                                if '+' == sent:
                                    date_category_reviews[date][display_name]['positive'] += 1
    
    # Convert the collected data to pandas DataFrame for easier manipulation
    date_entries = []
    
    for date_str, categories in date_category_reviews.items():
        date_obj = datetime.strptime(date_str, date_format)
        
        for category, data in categories.items():
            # Set a default sentiment value of 0.5 (neutral) since we don't have reliable 
            date_entries.append({
                'date': date_obj,
                'category': category,
                'count': data['count'],
                'sentiment': data['positive'] / data['count']
            })
    
    if not date_entries:
        # No data to process
        return time_series_data
        
    # Create DataFrame
    df = pd.DataFrame(date_entries)
    
    # Determine frequency for grouping based on time_bucket
    # Standard Pandas frequencies
    if time_bucket in ['day', 'week', 'month', 'year']:
        freq_map = {
            'day': 'D',
            'week': 'W',
            'month': 'M',
            'year': 'Y'
        }
        df['date_group'] = df['date'].dt.to_period(freq_map[time_bucket])
    else:
        # Handle custom time buckets
        if time_bucket == '3month':
            # Group by 3-month periods
            df['date_group'] = df['date'].dt.to_period('M')
            df['date_group'] = df['date_group'].apply(lambda x: pd.Period(f"{x.year}-{((x.month-1)//3)*3+1}", freq='3M'))
        elif time_bucket == '6month':
            # Group by half-year periods
            df['date_group'] = df['date'].dt.to_period('M')
            df['date_group'] = df['date_group'].apply(lambda x: pd.Period(f"{x.year}-{((x.month-1)//6)*6+1}", freq='6M'))
        else:
            # Default to month
            df['date_group'] = df['date'].dt.to_period('M')
    
    df['date_str'] = df['date_group'].dt.start_time.dt.strftime(date_format)
    
    # For sentiment, we need to calculate the weighted average based on count
    # to ensure that times with more reviews have more influence on the sentiment
    grouped = df.groupby(['date_str', 'category']).agg({
        'count': 'sum',
        'sentiment': lambda x: sum(a * b for a, b in zip(x, df.loc[x.index, 'count'])) / sum(df.loc[x.index, 'count']) if sum(df.loc[x.index, 'count']) > 0 else 0.5
    }).reset_index()
    
    # Sort by date
    grouped = grouped.sort_values('date_str')
    
    # Prepare return structure
    all_dates = sorted(grouped['date_str'].unique())
    time_series_data['dates'] = all_dates
    
    for category in display_categories:
        category_df = grouped[grouped['category'] == category]
        
        # Initialize with zeros for all dates
        counts = [0] * len(all_dates)
        sentiments = [0.5] * len(all_dates)  # Default neutral sentiment
        
        # Fill in data where available
        for _, row in category_df.iterrows():
            date_idx = all_dates.index(row['date_str'])
            counts[date_idx] = row['count']
            sentiments[date_idx] = row['sentiment']
            
        time_series_data['category_data'][category]['counts'] = counts
        time_series_data['category_data'][category]['sentiment'] = sentiments
    
    return time_series_data 