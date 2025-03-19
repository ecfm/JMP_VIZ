import json
import os
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any

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
            
            # Check if result is in cache
            if key in cache:
                return cache[key]
            
            # Call the function with original arguments
            result = func(*args, **kwargs)
            
            # Store result in cache
            cache[key] = result
            
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
        key_part_num = len(category.split('|'))+2
    filtered_dict = defaultdict(dict)
    for path, sents_dict in path_to_sents_dict.items():
        if category == 'all' or path.startswith(category):
            key = '|'.join(path.split('|')[:key_part_num])
            filtered_dict[key] = merge_values(filtered_dict[key], sents_dict)
    return filtered_dict

def extract_review_highlight(review):
    """Extract review text and highlight information."""
    highlight, review_text = review.split('|||')
    if '<<<' in highlight:
        highlight_detail, highlight_reason = highlight.split('<<<')
        return review_text, highlight_detail, highlight_reason
    else:
        return review_text, highlight, None

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
                            review_text, highlight_detail, highlight_reason = extract_review_highlight(review)
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
def get_cached_dict(type: str, category: str, search_query: str = '', return_both: bool = False) -> Dict:
    """
    Get filtered dictionary from cache. Can return both sents and ids dicts for attr/attr_perf.
    """
    raw_dict = raw_dict_map[type]
    
    # Filter by category
    filtered_dict = filter_dict(raw_dict, category)
    
    # Only apply search query filter to sents dictionaries
    if search_query and search_query.strip() and not type.endswith('_ids'):
        filtered_dict = filter_dict_by_query(filtered_dict, search_query)
        
    # If we need both sents and ids dicts
    if return_both:
        if type == 'attr_perf_sents':
            ids_dict = filter_dict(attr_perf_path_to_ids_dict, category)
            # Filter ids_dict to match paths in filtered sents dict
            ids_dict = {path: ids for path, ids in ids_dict.items() 
                       if path in filtered_dict}
        elif type == 'attr_sents':
            ids_dict = filter_dict(attr_path_to_ids_dict, category)
            # Filter ids_dict to match paths in filtered sents dict
            ids_dict = {path: ids for path, ids in ids_dict.items() 
                       if path in filtered_dict}
        else:
            raise ValueError(f"Cannot return both dicts for type: {type}")
            
        return filtered_dict, ids_dict
        
    return filtered_dict

@custom_cached(cache=plot_data_cache)
def get_plot_data(plot_type, x_category='all', y_category='all', top_n_x=2, top_n_y=2, language='en', search_query=''):
    """Get data for plotting the correlation matrix."""
    # Convert slider values to integers
    top_n_x = int(top_n_x)
    top_n_y = int(top_n_y)
    
    if plot_type == 'use_attr_perf':
        x_path_to_sents_dict = get_cached_dict('use_sents', x_category, search_query)
        y_path_to_sents_dict, y_path_to_ids_dict = get_cached_dict(
            'attr_perf_sents', y_category, search_query, return_both=True
        )
        title_key = 'use_attr_perf_title'
    else:  # perf_attr
        x_path_to_sents_dict = get_cached_dict('perf_sents', x_category, search_query)
        y_path_to_sents_dict, y_path_to_ids_dict = get_cached_dict(
            'attr_sents', y_category, search_query, return_both=True
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

    # Create clean text without percentages for lookups
    if x_category == 'all':
        x_text = top_x_paths
    else:
        x_text = [path[len(x_category)+1:] for path in top_x_paths]
    if y_category == 'all':
        y_text = top_y_paths
    else:
        y_text = [path[len(y_category)+1:] for path in top_y_paths]
    
    x_display = [f"({perc:.1f}%) {txt}" for txt, perc in zip(x_text, x_percentages)]
    y_display = [f"{txt} ({perc:.1f}%)" for txt, perc in zip(y_text, y_percentages)]
    
    return matrix, sentiment_matrix, review_matrix, x_text, x_display, y_text, y_display, title_key

def filter_dict_by_query(path_to_sents_dict: Dict, search_query: str) -> Dict:
    """Filter path_to_sents_dict by removing reviews that don't match the query"""
    if not search_query or not search_query.strip():
        return path_to_sents_dict
    
    # Create filter function once to reuse across all reviews
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
        filter_function = eval(filter_code, {"__builtins__": {}})
        
        # Define an efficient filtering function that uses our compiled filter
        def filter_reviews(reviews):
            if reviews is None:
                return []
            return [review for review in reviews if filter_function(review.lower())]
        
    except Exception as e:
        print(f"Search query error: {str(e)}")
        print(f"Filter expression was: {filter_expr if 'filter_expr' in locals() else 'not created'}")
        # Fall back to returning all reviews if filter creation fails
        return path_to_sents_dict
    
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
                    matching_reviews = filter_reviews(reviews)
                    if matching_reviews:  # Only keep if there are matching reviews
                        filtered_rids[rid] = matching_reviews
                if filtered_rids:  # Only keep sentiment if there are matching reviews
                    filtered_sents[sent] = filtered_rids
            elif isinstance(val, list):
                reviews = val
                matching_reviews = filter_reviews(reviews)
                if matching_reviews:
                    filtered_sents[sent] = matching_reviews

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

def get_options(value, top_paths):
    """Generate dropdown options for category selection."""
    levels = [{'label': 'All [Level 0]', 'value': 'all'}]
    if value != 'all':
        parts = value.split('|')
        for i, part in enumerate(parts):
            current_val = '|'.join(parts[:i+1])
            levels.append({'label': f'{current_val} [Level {i+1}]', 'value': current_val})
        remained_paths = [{'label': path+f' [Level {len(parts)+1}]', 'value': path} for path in top_paths if (path.startswith(value) and path != value)]
        if len(remained_paths) > 0:
            return levels + remained_paths
        else:
            return levels
    else:
        return levels + [{'label': path+' [Level 1]', 'value': path} for path in top_paths]

# Use our custom caching decorator instead of the built-in @cached
@custom_cached(cache=plot_data_cache)
def get_bar_chart_data(categories, zoom_category=None, language='en', search_query=''):
    """
    Get data for bar chart visualization.
    
    Args:
        categories: List of categories to include (usage, attribute, performance)
        zoom_category: Optional category to zoom in on (e.g., "usage|category1")
        language: Language for translations
        search_query: Optional search query to filter results
    
    Returns:
        Tuple containing (categories, counts, sentiment_ratios, review_data, colors, title_key)
    """
    result_categories = []
    result_counts = []
    result_sentiment_ratios = []
    result_review_data = []
    result_colors = []
    
    # Treat empty string as None
    if zoom_category == '':
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
            category_path = zoom_category[len(mapping['prefix']):]
            category_dict = get_cached_dict(dict_type, category_path, search_query)
        else:
            # Otherwise get all categories
            category_dict = get_cached_dict(dict_type, 'all', search_query)
        
        # Count mentions for each category
        for category_path, sents_dict in category_dict.items():
            # Get the display name for the category
            if zoom_category:
                # When zoomed in, show full subcategory names
                display_name = prefix + category_path.split('|')[-1]
            else:
                # Otherwise show the top-level category
                display_name = prefix + category_path.split('|')[0]
                
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
                            review_text, highlight, reason = extract_review_highlight(review)
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
                elif isinstance(reason_rids, list):
                    # It's a list of reviews directly
                    if reason_rids is None:
                        continue
                    for review in reason_rids:
                        review_text, highlight, reason = extract_review_highlight(review)
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
            
            # Only include categories with reviews
            if total_reviews:
                count = len(total_reviews)
                sentiment_ratio = len(positive_reviews) / count if count > 0 else 0
                
                # Check if category already exists in results (for aggregation)
                try:
                    idx = result_categories.index(display_name)
                    # If exists, add to count and update sentiment ratio
                    result_counts[idx] += count
                    positive_count = result_sentiment_ratios[idx] * result_counts[idx]
                    positive_count += len(positive_reviews)
                    result_sentiment_ratios[idx] = positive_count / result_counts[idx]
                    result_review_data[idx].extend(review_highlights)
                except ValueError:
                    # If not exists, add as new category
                    result_categories.append(display_name)
                    result_counts.append(count)
                    result_sentiment_ratios.append(sentiment_ratio)
                    result_review_data.append(review_highlights)
                    result_colors.append(color)
    
    # Sort by count in descending order
    sorted_indices = np.argsort(result_counts)[::-1]
    result_categories = [result_categories[i] for i in sorted_indices]
    result_counts = [result_counts[i] for i in sorted_indices]
    result_sentiment_ratios = [result_sentiment_ratios[i] for i in sorted_indices]
    result_review_data = [result_review_data[i] for i in sorted_indices]
    result_colors = [result_colors[i] for i in sorted_indices]
    
    return result_categories, result_counts, result_sentiment_ratios, result_review_data, result_colors, 'bar_chart_title' 