import dash
from dash import html
import json
import re

from config import TRANSLATIONS, SEARCH_EXAMPLES
from data import get_cached_dict

def get_search_examples_html(language):
    """Return search examples HTML for the given language."""
    return [
        html.P(
            line, 
            style={
                'marginBottom': '4px',
                'fontStyle': 'italic' if 'syntax' in line.lower() or '语法' in line else 'normal'
            }
        ) for line in SEARCH_EXAMPLES[language]
    ]

def normalize_search_query(query: str) -> str:
    """
    Normalize the search query for consistent processing and display.
    - Preserve boolean operators (&, |) and parentheses.
    - Handle quoted phrases correctly.
    - Remove excess whitespace but maintain the logical structure.
    """
    if not query:
        return ""

    # Clean up the query for display purposes, but preserve logical operators
    # First, ensure spaces around operators & and | for tokenization, but don't replace them
    display_query = query
    display_query = re.sub(r'([&|()])', r' \1 ', display_query)
    display_query = re.sub(r'\s+', ' ', display_query).strip()
    
    # Handle quoted phrases
    quoted_phrases = re.findall(r'"([^"]+)"', display_query)
    for phrase in quoted_phrases:
        # Ensure quoted phrases remain exactly as they are
        display_query = display_query.replace(f'"{phrase}"', f'"{phrase}"')
    
    return display_query


def logic_toggle_search_help(n_clicks, current_style):
    """Logic for toggling the search help tooltip visibility."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
            
    new_style = dict(current_style)
    if new_style.get('display') == 'none':
        new_style['display'] = 'block'
    else:
        new_style['display'] = 'none'
            
    return new_style


def logic_update_search_results_info(n_clicks, language, date_slider_value, search_query, plot_type, x_value, y_value, date_filter_storage):
    """Logic for displaying information about search results."""
    if n_clicks == 0 or not search_query or not search_query.strip():
        return ""
            
    date_range = json.loads(date_filter_storage) if date_filter_storage else {"start_date": None, "end_date": None}
    start_date = date_range.get("start_date")
    end_date = date_range.get("end_date")
    
    try:
        unique_review_ids = set()
        
        if plot_type == 'bar_chart':
            categories = ['usage', 'attribute', 'performance']
            for category_type in categories:
                dict_type = {'usage': 'use_sents', 'attribute': 'attr_sents', 'performance': 'perf_sents'}[category_type]
                filtered_dict = get_cached_dict(dict_type, 'all', search_query, start_date, end_date)
                for path, id_dict in filtered_dict.items():
                    for sent, val in id_dict.items():
                         if isinstance(val, dict):
                             unique_review_ids.update(val.keys())

        else:
            if plot_type == 'use_attr_perf':
                x_dict_type = 'use_sents'
                y_dict_type = 'attr_perf_sents'
            else:  # perf_attr
                x_dict_type = 'perf_sents'
                y_dict_type = 'attr_sents'
                    
            x_dict = get_cached_dict(x_dict_type, x_value or 'all', search_query, start_date, end_date)
            y_dict = get_cached_dict(y_dict_type, y_value or 'all', search_query, start_date, end_date)
                
            for dict_to_process in [x_dict, y_dict]:
                 for path, id_dict in dict_to_process.items():
                     for sent, val in id_dict.items():
                         if isinstance(val, dict):
                            unique_review_ids.update(val.keys())

        total_reviews = len(unique_review_ids)
        
        if total_reviews > 0:
            normalized_query = normalize_search_query(search_query)
            return TRANSLATIONS[language]['search_results'].format(total_reviews, normalized_query)
        else:
            return TRANSLATIONS[language].get('search_no_results', "No reviews found matching your search criteria.") # Use translation key
            
    except Exception as e:
        print(f"Error counting search results: {str(e)}")
        # Use translation key for error message
        return TRANSLATIONS[language].get('search_error', "Error processing search query.") 