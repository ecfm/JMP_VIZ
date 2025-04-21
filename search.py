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
    Normalize the search query for consistent processing.
    - Lowercase the query.
    - Split by logical operators (AND, OR, NOT) while preserving them.
    - Remove extra whitespace around terms and operators.
    - Reconstruct the query.
    - Handle quoted phrases correctly.
    """
    if not query:
        return ""

    # Regex to split by AND, OR, NOT, keeping delimiters, and also capturing quoted phrases
    # It captures:
    # 1. Quoted phrases ("...")
    # 2. Logical operators (AND, OR, NOT) surrounded by spaces (or start/end of string)
    # 3. Individual words/terms
    # 4. Parentheses (added capturing group)
    pattern = r'("[^"]+")|\b(AND|OR|NOT)\b|\b([^\s"()]+)\b|([()])'
    
    tokens = re.findall(pattern, query, re.IGNORECASE)
    
    processed_tokens = []
    for quoted, operator, word, parenthesis in tokens:
        if quoted:
            # Keep quoted phrases as is, but maybe lowercase content inside?
            # For now, keep as is to preserve potential case sensitivity if needed later
            processed_tokens.append(quoted)
        elif operator:
            # Uppercase operators
            processed_tokens.append(operator.upper())
        elif word:
            # Lowercase individual words
            processed_tokens.append(word.lower())
        elif parenthesis:
             # Keep parentheses
            processed_tokens.append(parenthesis)

    # Reconstruct the query, ensuring single spaces between tokens
    return ' '.join(token for token in processed_tokens if token)


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