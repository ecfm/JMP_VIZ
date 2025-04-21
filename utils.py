import plotly.colors
import plotly.express as px
from dash import html, dcc
import re
from config import type_colors, TRANSLATIONS
import numpy as np

def ratio_to_rgb(ratio):
    """Convert a ratio value (0-1) to an RGB color."""
    ratio = max(0.00001, min(ratio, 0.99999))
    rgba = plotly.colors.sample_colorscale(px.colors.diverging.RdBu, ratio)[0]
    return rgba

def get_options(language, value, top_paths, top_display_paths):
    """Generate dropdown options for category selection."""
    levels = [{'label': TRANSLATIONS[language]['all_level_0'], 'value': 'all'}]
    if value != 'all':
        display_parts = [format_category_display(value, language)]
        parts = value.split('|')
        for i, part in enumerate(parts):
            current_val = '|'.join(parts[:i+1])
            display_current_val = format_category_display(current_val, language)
            prefix = '--' * (i+1)
            levels.append({'label': f'{prefix} {display_current_val} [L{i+1}]', 'value': current_val})
        
        # Format remaining paths
        remained_paths = []
        for path, display_path in zip(top_paths, top_display_paths):
            prefix = '--' * (len(parts)+1)
            remained_paths.append({
                'label': f'{prefix} {display_path} [L{len(parts)+1}]', 
                'value': path
            })
            
        return levels + remained_paths
    else:
        # Format top-level paths
        top_level_paths = []
        for path, display_path in zip(top_paths, top_display_paths):
            top_level_paths.append({
                'label': f'-- {display_path} [L1]',
                'value': path
            })
            
        return levels + top_level_paths

def format_category_display(category, language):
    """
    Format category display text based on language.
    For Chinese (zh): Remove content within parentheses
    For English (en): Extract and keep only content within parentheses, handle hierarchy
    For other languages: Remove content within parentheses
    
    Args:
        category: The category string to format
        language: The current language setting
        
    Returns:
        Formatted category string
    """
    # Extract prefix if present (U:, A:, P:)
    prefix = ""
    for p in type_colors.keys():
        if category.startswith(p):
            prefix = p
            category = category[len(p):]  # Remove prefix temporarily
            break
            
    if language == 'en':
        # For English, handle both hierarchy and parentheses
        if '|' in category:
            # If there's a hierarchy, keep everything after the last '|'
            parts = category.split('|')
            if len(parts) > 1:
                # Extract content within parentheses from the last part
                last_part = parts[-1]
                parentheses_match = re.search(r'\s*\(([^)]*)\)', last_part)
                if parentheses_match:
                    formatted = parentheses_match.group(1)
                else:
                    formatted = last_part
            else:
                formatted = category
        else:
            # For non-hierarchical categories, extract content within parentheses if present
            parentheses_match = re.search(r'\s*\(([^)]*)\)', category)
            if parentheses_match:
                formatted = parentheses_match.group(1)
            else:
                formatted = category
    else:
        # For Chinese and other languages, remove content within parentheses
        formatted = re.sub(r'\s*\([^)]*\)', '', category)
    
    # Reapply the prefix if it was present
    if prefix:
        return prefix + formatted
    return formatted

def create_search_box(language='en'):
    """Create a search box component with the given language."""
    from config import TRANSLATIONS
    
    return html.Div([
        html.Div([
            html.Div([
                dcc.Input(
                    id='search-input',
                    type='text',
                    placeholder=TRANSLATIONS[language]['search_placeholder'],
                    n_submit=0,  # Add this for Enter key handling
                    style={
                        'width': '70%', 
                        'marginRight': '10px',
                        'padding': '8px',
                        'fontSize': '14px'
                    }
                ),
                html.Button(
                    TRANSLATIONS[language]['search'], 
                    id='search-button', 
                    n_clicks=0,
                    style={
                        'padding': '8px 16px',
                        'fontSize': '14px',
                        'cursor': 'pointer',
                        'marginRight': '10px'
                    }
                ),
                html.Div([
                    html.Button(
                        "?",
                        id='search-help-button',
                        style={
                            'padding': '8px 12px',
                            'fontSize': '14px',
                            'cursor': 'pointer',
                            'borderRadius': '50%',
                            'fontWeight': 'bold',
                            'backgroundColor': '#c5c5c5',
                            'border': '1px solid #ccc'
                        }
                    ),
                    html.Div(
                        id='search-help-tooltip',
                        style={
                            'display': 'none',
                            'position': 'absolute',
                            'backgroundColor': 'white',
                            'border': '1px solid #ccc',
                            'padding': '15px',
                            'borderRadius': '5px',
                            'zIndex': '1000',
                            'boxShadow': '0px 2px 10px rgba(0,0,0,0.2)',
                            'maxWidth': '400px',
                            'width': '400px',
                            'fontSize': '12px',
                            'right': '0',
                            'marginTop': '5px'
                        }
                    )
                ], style={'display': 'inline-block', 'position': 'relative'})
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Div(id='search-results-info', style={
                'marginTop': '10px',
                'fontSize': '14px',
                'color': '#444444',  # Darker grey text
                'display': 'inline-block',  # Make div only as wide as content
                'backgroundColor': '#FFEB3B',  # Bright yellow background
                'padding': '0 4px',  # Add small horizontal padding
            })
        ], style={'marginBottom': '20px'})
    ])
    
def get_size_legend_translations(language):
    """Return size legend translations for the given language."""
    return {
        'en': {
            'explanation': "Square size is proportional to number of mentions"
        },
        'zh': {
            'explanation': "方块大小与提及次数成正比"
        }
    }[language]

def get_hover_translations(language):
    """Return hover translations for the given language."""
    from config import TRANSLATIONS
    
    return {
        'hover_x': TRANSLATIONS[language]['hover_x'],
        'hover_y': TRANSLATIONS[language]['hover_y'],
        'hover_count': TRANSLATIONS[language]['hover_count'],
        'hover_satisfaction': TRANSLATIONS[language]['hover_satisfaction']
    }

def get_proportional_size(value, max_val, min_val):
    """Calculate a square size that makes area proportional to the value."""
    if value == 0:
        return 0
    
    # For a square, area = size^2
    # To make area proportional to value, size should be proportional to sqrt(value)
    
    # Add small epsilon to avoid division by zero
    epsilon = 0.000001
    
    # Calculate sqrt(value)/sqrt(max_val) * 0.8
    # The 0.8 factor is to ensure squares don't overlap too much
    return 0.8 * np.sqrt(value) / (np.sqrt(max_val) + epsilon) 