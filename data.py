import json
import os
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from config import (
    get_result_dir, 
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

# Global dictionary to store raw data
raw_dict_map = {}

def load_data_files(category='Cables'):
    """Load data files for the specified category."""
    result_dir = get_result_dir(category)
    return {
        'use_sents': json.load(open(os.path.join(result_dir, 'use_path_to_sents_dict.json'))),
        'attr_perf_ids': json.load(open(os.path.join(result_dir, 'attr_perf_path_to_ids_dict.json'))),
        'attr_perf_sents': json.load(open(os.path.join(result_dir, 'attr_perf_path_to_sents_dict.json'))),
        'perf_sents': json.load(open(os.path.join(result_dir, 'perf_path_to_sents_dict.json'))),
        'attr_ids': json.load(open(os.path.join(result_dir, 'attr_path_to_ids_dict.json'))),
        'attr_sents': json.load(open(os.path.join(result_dir, 'attr_path_to_sents_dict.json')))
    }

def update_raw_dict_map(category='Cables'):
    """Update the raw dictionary map with data for the specified category."""
    global raw_dict_map
    raw_dict_map = load_data_files(category)

# Initial load with default category
update_raw_dict_map()

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
    for path, data_dict in path_to_sents_dict.items():
        if category == 'all' or path.startswith(category):
            key = '|'.join(path.split('|')[:key_part_num])
            filtered_dict[key] = merge_values(filtered_dict[key], data_dict)
    return filtered_dict

def parse_ngrams(ngram_str: Optional[str]) -> List[Dict[str, any]]:
    """Parse ngram string like 'ngram1<pos1:pos2>|ngram2<pos3:pos4>'"""
    ngrams = []
    if not ngram_str:
        return ngrams
    items = ngram_str.split('|')
    for item in items:
        if item:
            match = re.search(r'([^<]+)<(\d+):(\d+)>', item)
            if match:
                ngrams.append({
                    'text': match.group(1),
                    'start': int(match.group(2)),
                    'end': int(match.group(3))
                })
    return ngrams

def extract_text_and_ngrams(text_with_ngrams: str) -> Tuple[str, List[Dict[str, any]]]:
    """Separate text and ngrams from a string like 'some text###ngram<1:5>'"""
    if '###' in text_with_ngrams:
        text, ngram_part = text_with_ngrams.split('###', 1)
        ngrams = parse_ngrams(ngram_part)
        return text.strip(), ngrams
    else:
        return text_with_ngrams.strip(), []

def extract_review_info(review_string: str) -> Optional[Dict]:
    """
    Extract structured information from the new review format.

    Args:
        review_string: Full review string in the new format.

    Returns:
        Dictionary containing extracted parts or None if format is invalid.
        Keys: 'highlight_detail_text', 'highlight_detail_ngrams',
              'highlight_reason_text', 'highlight_reason_ngrams',
              'product_id', 'star_rating', 'date', 'review_title',
              'review_content', 'review_ngrams', 'full_review_text', 'raw_review'
    """
    try:
        # Split highlight from the rest
        highlight_part, rest_part = review_string.split('|||', 1)

        # Process highlight part (detail and optional reason)
        highlight_detail_text = ""
        highlight_detail_ngrams = []
        highlight_reason_text = None
        highlight_reason_ngrams = []

        if '<<<' in highlight_part:
            detail_part, reason_part = highlight_part.split('<<<', 1)
            highlight_detail_text, highlight_detail_ngrams = extract_text_and_ngrams(detail_part)
            # Reason text might be quoted, remove quotes if present
            if reason_part.startswith('"') and reason_part.endswith('"'):
                reason_part = reason_part[1:-1]
            highlight_reason_text, highlight_reason_ngrams = extract_text_and_ngrams(reason_part)
        else:
            highlight_detail_text, highlight_detail_ngrams = extract_text_and_ngrams(highlight_part)

        # Process the rest of the string (review text and review ngrams)
        review_text_part, *review_ngram_part_list = rest_part.split('###', 1)
        review_ngrams = parse_ngrams(review_ngram_part_list[0]) if review_ngram_part_list else []

        # Extract product_id, rating, date, and full text from review_text_part
        # Example: "B0CXYZABC ★★★☆☆ [2023-10-26] <b>Title</b> Content"
        # Match product ID (non-space chars at start)
        match = re.match(r'(\S+)\s+(.*)', review_text_part.strip())
        if not match:
            # print(f"Warning: Could not parse product ID from: {review_text_part[:50]}...")
            return None
        product_id = match.group(1)
        remaining_text = match.group(2)

        # Match star rating (★ symbols)
        rating_match = re.match(r'([★☆]+)\s+(.*)', remaining_text)
        if not rating_match:
             # print(f"Warning: Could not parse star rating from: {remaining_text[:50]}...")
            return None
        star_rating = rating_match.group(1)
        remaining_text = rating_match.group(2)

        # Match date
        date_match = re.match(r'\[(\d{4}-\d{2}-\d{2})\]\s+(.*)', remaining_text, re.DOTALL)
        if not date_match:
            # print(f"Warning: Could not parse date from: {remaining_text[:50]}...")
            return None
        date = date_match.group(1)
        full_review_text = date_match.group(2).strip() # This is "<b>Title</b> Content"

        # Extract title and content
        title_match = re.search(r'<b>(.*?)</b>(.*)', full_review_text, re.DOTALL)
        if title_match:
            review_title = title_match.group(1).strip()
            review_content = title_match.group(2).strip()
        else:
            # Assume no title, content is the whole text
            review_title = ""
            review_content = full_review_text # Keep potential leading/trailing spaces if no title tag

        return {
            'highlight_detail_text': highlight_detail_text,
            'highlight_detail_ngrams': highlight_detail_ngrams,
            'highlight_reason_text': highlight_reason_text,
            'highlight_reason_ngrams': highlight_reason_ngrams,
            'product_id': product_id,
            'star_rating': star_rating,
            'date': date,
            'review_title': review_title,
            'review_content': review_content,
            'review_ngrams': review_ngrams,
            'full_review_text': full_review_text,
            'raw_review': review_string # Keep the original string
        }

    except Exception:
        # print(f"Error parsing review string: {review_string[:100]}... Error: {e}")
        return None

def create_correlation_matrix(x_path_to_sents_dict, y_path_to_ids_dict, y_path_to_sents_dict):
    """Create correlation matrix between x and y categories using review dictionaries."""
    # Determine matrix dimensions
    num_y_cats = len(y_path_to_ids_dict)
    num_x_cats = len(x_path_to_sents_dict)

    # Initialize matrices
    matrix = np.zeros((num_y_cats, num_x_cats))
    sentiment_matrix = np.zeros((num_y_cats, num_x_cats))
    # Use dtype=object for storing lists of dictionaries
    review_dict_matrix = np.empty((num_y_cats, num_x_cats), dtype=object)

    # Initialize each cell in review_dict_matrix with an empty list
    for i in range(num_y_cats):
        for j in range(num_x_cats):
            review_dict_matrix[i, j] = []

    # Create a mapping from y_path to its index for quick lookup
    y_path_to_index = {path: i for i, path in enumerate(y_path_to_ids_dict.keys())}

    # Iterate through X categories (paths and their sentiment/review data)
    for j, (x_path, x_sents_data) in enumerate(x_path_to_sents_dict.items()):
        # Iterate through sentiments/reasons within the X category data
        for sent, reason_rids_or_reviews in x_sents_data.items():
            # Check if the value is a dictionary {rid: [review_dict]} or list [review_dict]
            if isinstance(reason_rids_or_reviews, dict):
                # Structure: {rid: [review_dict, ...]}
                for rid, review_dicts_list in reason_rids_or_reviews.items():
                    # Find corresponding Y categories that mention this rid
                    for y_path, y_ids in y_path_to_ids_dict.items():
                        if rid in y_ids:
                            # Get the index for the Y category
                            i = y_path_to_index.get(y_path)
                            if i is not None:
                                # Process each review dictionary for this co-occurrence
                                if review_dicts_list:
                                    for review_info in review_dicts_list:
                                        if review_info: # Ensure review_info is not None
                                            # Append review dictionary to the cell
                                            review_dict_matrix[i, j].append(review_info)
                                            # Increment total count
                                            matrix[i, j] += 1
                                            # Check sentiment for positive count
                                            stars = review_info.get('star_rating', '').count('★')
                                            if stars >= 4:
                                                sentiment_matrix[i, j] += 1 # Sum positive counts first
            # TODO: Handle case where reason_rids_or_reviews is just a list of review dicts?
            # This might occur if the structure simplifies, but filter_dict_by_query seems
            # to preserve the rid structure. Add handling if necessary based on observed data.
            elif isinstance(reason_rids_or_reviews, list):
                 print(f"Warning: Unexpected list structure found for {x_path} -> {sent} in create_correlation_matrix. Add handling if needed.")

    # Final pass to calculate sentiment ratio
    for i in range(num_y_cats):
        for j in range(num_x_cats):
            total_count = matrix[i, j]
            if total_count > 0:
                # Calculate ratio from summed positive counts
                sentiment_matrix[i, j] = sentiment_matrix[i, j] / total_count
            else:
                sentiment_matrix[i, j] = 0.5 # Default to neutral if no reviews

    return matrix, sentiment_matrix, review_dict_matrix

@custom_cached(cache=path_dict_cache)
def get_cached_dict(plot_type: str, category: str, search_query: str = '', start_date: str = None, end_date: str = None, return_both: bool = False) -> Dict:
    """
    Get filtered dictionary from cache. Can return both sents and ids dicts for attr/attr_perf.
    """
    if plot_type not in raw_dict_map:
        raise KeyError(f"Invalid plot type: {plot_type}. Available types: {list(raw_dict_map.keys())}")
        
    raw_dict = raw_dict_map[plot_type]
    
    # Filter by category
    filtered_dict = filter_dict(raw_dict, category)
    
    # Only apply search query filter to sents dictionaries
    if (search_query and search_query.strip() and not plot_type.endswith('_ids')) or start_date or end_date:
        filtered_dict = filter_dict_by_query(filtered_dict, search_query, start_date, end_date)
        
    # If we need both sents and ids dicts
    if return_both:
        if plot_type == 'attr_perf_sents':
            ids_dict = filter_dict(raw_dict_map['attr_perf_ids'], category)
            # Filter ids_dict to match paths in filtered sents dict
            ids_dict = {path: ids for path, ids in ids_dict.items() 
                       if path in filtered_dict}
        elif plot_type == 'attr_sents':
            ids_dict = filter_dict(raw_dict_map['attr_ids'], category)
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

    matrix, sentiment_matrix, review_dict_matrix = create_correlation_matrix(
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
    review_dict_matrix = review_dict_matrix[top_y_indices, :][:, top_x_indices] # Slice dict matrix
    
    return matrix, sentiment_matrix, review_dict_matrix, top_x_paths, x_percentages, top_y_paths, y_percentages, title_key

def filter_dict_by_query(path_to_sents_dict: Dict, search_query: str, start_date: str = None, end_date: str = None) -> Dict:
    """
    Filter path_to_sents_dict by removing reviews that don't match the query or date range.
    Stores processed review dictionaries instead of raw strings.
    """
    if (not search_query or not search_query.strip()) and start_date is None and end_date is None:
        # If no filter, still parse reviews to ensure consistent output format
        processed_dict = {}
        for path, sents_dict in path_to_sents_dict.items():
            processed_sents = {}
            if isinstance(sents_dict, list): # Handle IDs dict case? No, only sents dicts are passed here now.
                 processed_dict[path] = sents_dict # Pass IDs through directly
                 continue

            for sent, val in sents_dict.items():
                if isinstance(val, dict): # rid -> reviews map
                    processed_rids = {}
                    for rid, reviews in val.items():
                        processed_reviews = []
                        if reviews:
                            for review_str in reviews:
                                review_info = extract_review_info(review_str)
                                if review_info:
                                    processed_reviews.append(review_info)
                        if processed_reviews:
                             processed_rids[rid] = processed_reviews
                    if processed_rids:
                        processed_sents[sent] = processed_rids
                elif isinstance(val, list): # list of reviews
                    processed_reviews = []
                    if val:
                        for review_str in val:
                             review_info = extract_review_info(review_str)
                             if review_info:
                                 processed_reviews.append(review_info)
                    if processed_reviews:
                         processed_sents[sent] = processed_reviews
            if processed_sents:
                processed_dict[path] = processed_sents
        return processed_dict

    # Function to extract and filter review information
    def extract_and_filter_review(review_string, query_filter=None):
        """
        Extract info from a review string and apply filters

        Args:
            review_string: The raw review string to process
            query_filter: Optional function to filter by query text

        Returns:
            Processed review dictionary or None if filtered out
        """
        # Extract all review information
        review_info = extract_review_info(review_string)

        if not review_info:
            return None

        # Apply date filter
        date = review_info['date']
        if start_date is not None or end_date is not None:
            if date is None or (start_date and date < start_date) or (end_date and date > end_date):
                return None

        # Apply search query filter if provided
        # Use full_review_text for filtering
        if query_filter and not query_filter(review_info['full_review_text'].lower()):
            return None

        # Return the processed review dictionary
        return review_info

    # Function to process a list of reviews
    def process_reviews(reviews_list, query_filter=None):
        """Process review strings: extract info and apply filters"""
        if reviews_list is None:
            return []

        processed_review_dicts = []
        for review_str in reviews_list:
            processed_review = extract_and_filter_review(review_str, query_filter)
            if processed_review:  # Only keep reviews that pass filters
                processed_review_dicts.append(processed_review)

        return processed_review_dicts

    # Create query filter function if search query is provided
    query_filter = None
    if search_query and search_query.strip():
        try:
            # Convert operators and clean up query
            query = search_query.lower().strip()
            
            # Add spaces around parentheses to ensure proper tokenization
            query = re.sub(r'\(', ' ( ', query)
            query = re.sub(r'\)', ' ) ', query)
            query = re.sub(r'\s+', ' ', query).strip()
            
            query = re.sub(r'\s*&\s*', ' and ', query)
            query = re.sub(r'\s*\|\s*', ' or ', query)
            
            # Parse terms in quotes as single terms
            quoted_terms = re.findall(r'"([^"]+)"', query)
            for term in quoted_terms:
                # Replace the quoted term with a placeholder that won't be split
                placeholder = f"__QUOTED_{term.replace(' ', '_')}__"
                query = query.replace(f'"{term}"', placeholder)
            
            # Split by spaces but preserve parentheses and operators
            tokens = re.findall(r'__QUOTED_[^_]+__|[()]|\band\b|\bor\b|\S+', query)
            
            # Convert terms to regex patterns
            terms = []
            current_terms = []
            paren_groups = []
            
            for token in tokens:
                if token == '(':
                    # Save current terms if any before starting a new group
                    if current_terms:
                        if len(current_terms) > 1:
                            terms.append(f"({' or '.join(current_terms)})")
                        else:
                            terms.append(current_terms[0])
                        current_terms = []
                    # Push a new group onto the stack
                    paren_groups.append(terms)
                    terms = []
                elif token == ')':
                    # Handle any pending terms in current group
                    if current_terms:
                        if len(current_terms) > 1:
                            terms.append(f"({' or '.join(current_terms)})")
                        else:
                            terms.append(current_terms[0])
                        current_terms = []
                    
                    # Pop the outer group and add current group to it
                    if not paren_groups:
                        raise ValueError("Unmatched closing parenthesis")
                    
                    # Join current terms with OR if multiple
                    current_expr = f"({' or '.join(terms)})" if len(terms) > 1 else terms[0] if terms else ""
                    
                    # Restore outer group
                    terms = paren_groups.pop()
                    if current_expr:
                        terms.append(current_expr)
                elif token in ('and', 'or'):
                    # Handle any pending terms before the operator
                    if current_terms:
                        if len(current_terms) > 1:
                            terms.append(f"({' or '.join(current_terms)})")
                        else:
                            terms.append(current_terms[0])
                        current_terms = []
                    terms.append(token)
                else:
                    # Handle quoted terms and regular terms
                    if token.startswith('__QUOTED_') and token.endswith('__'):
                        clean_term = token[9:-2].replace('_', ' ')
                        current_terms.append(f"'{clean_term}' in review_text")
                    else:
                        current_terms.append(f"'{token}' in review_text")
            
            # Handle any remaining terms
            if current_terms:
                if len(current_terms) > 1:
                    terms.append(f"({' or '.join(current_terms)})")
                else:
                    terms.append(current_terms[0])
            
            # Check for unmatched opening parentheses
            if paren_groups:
                raise ValueError("Unmatched opening parenthesis")
            
            # Create the filter expression
            filter_expr = ' '.join(terms)
            
            # Handle edge cases with parentheses
            filter_expr = re.sub(r'\(\s+', '(', filter_expr)
            filter_expr = re.sub(r'\s+\)', ')', filter_expr)
            
            # Validate the expression is not empty
            if not filter_expr.strip():
                raise ValueError("Empty search expression")
            
            # Create a compiled filter function to avoid re-evaluation
            try:
                filter_code = compile(f"lambda review_text: {filter_expr}", "<string>", "eval")
                query_filter = eval(filter_code, {"__builtins__": {}})
            except Exception as e:
                print(f"Search query error: {str(e)}")
                print(f"Filter expression was: {filter_expr if 'filter_expr' in locals() else 'not created'}")
                query_filter = None # Fall back to just date filtering
            
        except Exception as e:
            print(f"Search query error: {str(e)}")
            print(f"Filter expression was: {filter_expr if 'filter_expr' in locals() else 'not created'}")
            query_filter = None # Fall back to just date filtering
    
    # Now apply the filter to the dictionary structure
    filtered_dict = {}
    for path, sents_dict in path_to_sents_dict.items():
        # Handle case where value is a list (ids_dict case) - Should not happen if only sents dicts are passed
        if isinstance(sents_dict, list):
             filtered_dict[path] = sents_dict # Pass IDs through directly? Or should IDs be filtered too? Assume pass through for now.
             continue

        # Handle case where value is a dict (sents_dict case)
        filtered_sents = {}
        for sent, val in sents_dict.items():
            if isinstance(val, dict):
                rid_reviews_dict = val
                filtered_rids = {}
                for rid, reviews_list in rid_reviews_dict.items():
                    processed_review_dicts = process_reviews(reviews_list, query_filter) # Process list of strings
                    if processed_review_dicts:  # Only keep if there are matching reviews
                        filtered_rids[rid] = processed_review_dicts # Store list of dicts
                if filtered_rids:  # Only keep sentiment if there are matching reviews
                    filtered_sents[sent] = filtered_rids
            elif isinstance(val, list):
                reviews_list = val
                processed_review_dicts = process_reviews(reviews_list, query_filter) # Process list of strings
                if processed_review_dicts:
                    filtered_sents[sent] = processed_review_dicts # Store list of dicts

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
    def process_review(review_dict, # Now takes a dict
                       total_reviews, positive_reviews, review_highlights):
        """
        Process a single review dictionary and update the tracking collections

        Args:
            review_dict: The review dictionary from extract_review_info
            total_reviews: Set to track unique reviews (using full_review_text)
            positive_reviews: Set to track positive reviews (using full_review_text)
            review_highlights: List to collect review dictionaries
        """
        if not review_dict: # Skip if parsing failed
            return

        review_text_key = review_dict['full_review_text'] # Use full text as key
        total_reviews.add(review_text_key)

        # Determine sentiment from star rating
        stars = review_dict['star_rating'].count('★')
        is_positive = stars >= 4

        # Add review dictionary to highlights list
        # We also need sentiment info here for the caller function
        review_highlights.append({'sentiment': '+' if is_positive else ('-' if stars <=2 else '?'),
                                  'review_info': review_dict})

        # Track positive reviews
        if is_positive:
            positive_reviews.add(review_text_key)

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
            unique_reviews_text = set() # Track unique review texts
            positive_reviews_text = set() # Track unique positive review texts
            category_review_dicts_with_sentiment = [] # Collect dicts for this category

            # Process all reviews for this category
            for sent, reason_rids in sents_dict.items(): # 'sent' key might be obsolete now
                # Check if reason_rids is a dictionary or a list of review dicts
                if isinstance(reason_rids, dict):
                    # It's a dictionary with rid -> list of review dicts
                    for rid, review_dicts in reason_rids.items():
                        if review_dicts is None: continue
                        for review_dict in review_dicts:
                            process_review(review_dict, unique_reviews_text, positive_reviews_text, category_review_dicts_with_sentiment)
                elif isinstance(reason_rids, list):
                    # It's a list of review dicts directly
                    if reason_rids is None: continue
                    for review_dict in reason_rids:
                        process_review(review_dict, unique_reviews_text, positive_reviews_text, category_review_dicts_with_sentiment)

            # Only include categories with reviews
            if unique_reviews_text:
                count = len(unique_reviews_text)
                sentiment_ratio = len(positive_reviews_text) / count if count > 0 else 0.5 # Use 0.5 for neutral

                # Aggregate review dictionaries if category already exists
                if display_name in result_dict:
                    # If exists, add to count and update sentiment ratio
                    existing_count = result_dict[display_name]['count']
                    existing_pos_count = result_dict[display_name]['sentiment_ratio'] * existing_count

                    new_total_count = existing_count + count
                    # Estimate new positive count based on ratio of new reviews
                    new_pos_reviews_from_current = count * sentiment_ratio
                    new_pos_count = existing_pos_count + new_pos_reviews_from_current

                    result_dict[display_name]['count'] = new_total_count
                    result_dict[display_name]['sentiment_ratio'] = new_pos_count / new_total_count if new_total_count > 0 else 0.5
                    result_dict[display_name]['reviews'].extend(category_review_dicts_with_sentiment) # Append list of dicts
                else:
                    # If not exists, add as new category
                    result_dict[display_name] = {
                        'original_category': prefix + original_category,
                        'count': count,
                        'sentiment_ratio': sentiment_ratio,
                        'reviews': category_review_dicts_with_sentiment, # Store list of dicts
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
        review_data.append(item['reviews']) # review_data is now list of lists of dicts
        colors.append(item['color'])
    
    return display_categories, original_categories, counts, sentiment_ratios, review_data, colors, 'bar_chart_title'

def get_review_date_range():
    """Find the earliest and latest dates from all review dictionaries."""
    min_date = None
    max_date = None

    # Process all review dictionaries to find date range
    # Assuming raw_dict_map holds the original structure with raw strings
    for dict_type in ['use_sents', 'attr_perf_sents', 'perf_sents', 'attr_sents']:
        if dict_type not in raw_dict_map: continue
        path_to_sents_dict = raw_dict_map[dict_type]

        for path, sents_dict in path_to_sents_dict.items():
            for sent, val in sents_dict.items():
                reviews_list = []
                if isinstance(val, dict): # rid -> reviews map
                    for rid, reviews in val.items():
                        if reviews: reviews_list.extend(reviews)
                elif isinstance(val, list): # list of reviews
                    if val: reviews_list.extend(val)

                for review_str in reviews_list:
                    # Need to parse just the date quickly
                    date_match = re.search(r'\[(\d{4}-\d{2}-\d{2})\]', review_str)
                    date = date_match.group(1) if date_match else None
                    # # Alternative: Use full parsing (slower)
                    # review_info = extract_review_info(review_str)
                    # date = review_info['date'] if review_info else None

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
    Uses review dictionaries now.
    """
    from datetime import datetime
    import pandas as pd
        
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
        'category_data': {cat: {'counts': [], 'sentiment': []} for cat in display_categories},
        'time_bucket': time_bucket
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
        
        # Get the dictionary for this category type, applying filters
        # This should now return dicts with review_info dictionaries
        category_dict = get_cached_dict(dict_type, zoom_category if zoom_category else 'all',
                                        search_query, start_date, end_date)

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
            for sent, reason_rids in sents_dict.items(): # sent key might be irrelevant
                review_dicts_list = []
                if isinstance(reason_rids, dict):
                    # Dictionary with rid -> list of review dicts
                    for rid, review_dicts in reason_rids.items():
                        if review_dicts: review_dicts_list.extend(review_dicts)
                elif isinstance(reason_rids, list):
                    # List of review dicts directly
                    if reason_rids: review_dicts_list.extend(reason_rids)

                for review_info in review_dicts_list:
                    if not review_info: continue # Skip if parsing failed

                    date = review_info['date']
                    if date:
                        # Initialize date entry if needed
                        if date not in date_category_reviews:
                            date_category_reviews[date] = {}

                        # Initialize category for this date if needed
                        if display_name not in date_category_reviews[date]:
                            date_category_reviews[date][display_name] = {
                                'count': 0,
                                'positive': 0,
                                'total_sentiment_score': 0 # For weighted average
                            }

                        # Count review and track sentiment
                        stars = review_info['star_rating'].count('★')
                        is_positive = stars >= 4
                        sentiment_score = 1.0 if is_positive else (0.0 if stars <= 2 else 0.5)

                        date_category_reviews[date][display_name]['count'] += 1
                        if is_positive:
                            date_category_reviews[date][display_name]['positive'] += 1
                        date_category_reviews[date][display_name]['total_sentiment_score'] += sentiment_score


    # Convert the collected data to pandas DataFrame for easier manipulation
    date_entries = []
    
    for date_str, categories in date_category_reviews.items():
        date_obj = datetime.strptime(date_str, date_format)
        
        for category, data in categories.items():
            date_entries.append({
                'date': date_obj,
                'category': category,
                'count': data['count'],
                # Calculate average sentiment for this specific day/category entry
                'sentiment': data['total_sentiment_score'] / data['count'] if data['count'] > 0 else 0.5
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
    
    df['date_str'] = df['date_group'].dt.start_time.dt.strftime('%Y-%m')
    
    # For sentiment, we need to calculate the weighted average based on count
    # Group first, then calculate weighted average
    grouped_agg = df.groupby(['date_str', 'category']).agg(
        total_count=('count', 'sum'),
        # We need individual sentiments and counts to calculate weighted average
    ).reset_index()

    # Calculate weighted sentiment for each group
    sentiment_map = {}
    for name, group in df.groupby(['date_str', 'category']):
        weighted_sum = (group['sentiment'] * group['count']).sum()
        total_count = group['count'].sum()
        avg_sentiment = weighted_sum / total_count if total_count > 0 else 0.5
        sentiment_map[name] = avg_sentiment

    # Add weighted sentiment back to the aggregated dataframe
    grouped_agg['sentiment'] = grouped_agg.apply(lambda row: sentiment_map.get((row['date_str'], row['category']), 0.5), axis=1)
    grouped_agg = grouped_agg.rename(columns={'total_count': 'count'})


    # Sort by date
    grouped = grouped_agg.sort_values('date_str')

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