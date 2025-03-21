import plotly.colors
import plotly.express as px
from dash import html, dcc

def ratio_to_rgb(ratio):
    """Convert a ratio value (0-1) to an RGB color."""
    ratio = max(0.00001, min(ratio, 0.99999))
    rgba = plotly.colors.sample_colorscale(px.colors.diverging.RdBu, ratio)[0]
    return rgba

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
                    TRANSLATIONS[language].get('search', 'Search'), 
                    id='search-button', 
                    n_clicks=0,
                    style={
                        'padding': '8px 16px',
                        'fontSize': '14px',
                        'cursor': 'pointer'
                    }
                ),
            ]),
            html.Div(id='search-results-info', style={
                'marginTop': '10px',
                'fontSize': '14px',
                'color': '#444444',  # Darker grey text
                'display': 'inline-block',  # Make div only as wide as content
                'backgroundColor': '#FFEB3B',  # Bright yellow background
                'padding': '0 4px',  # Add small horizontal padding
            }),
            html.Div([
                html.Div(id='search-examples', style={
                    'marginTop': '8px', 
                    'fontSize': '12px',
                    'color': '#666'
                })
            ])
        ], style={'marginBottom': '20px'})
    ])
    
def get_width_legend_translations(language):
    """Return width legend translations for the given language."""
    return {
        'en': {
            'explanation': "Width is proportional to log(#reviews)"
        },
        'zh': {
            'explanation': "宽度与log(评论数量)成正比"
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

def get_log_width(value, max_val, min_val):
    """Calculate a logarithmic width for visualization."""
    import numpy as np
    
    if value == 0:
        return 0
    # Add 1 to avoid log(1) = 0
    log_val = np.log(value + 2)
    log_max = np.log(max_val + 1)
    log_min = np.log(min_val + 1)
    # Normalize to 0.8 max width
    return 0.8 * (value) / (max_val  + 0.000001)

def get_search_examples_html(language):
    """Return search examples HTML for the given language."""
    from config import SEARCH_EXAMPLES
    
    return [
        html.P(
            line, 
            style={
                'marginBottom': '4px',
                'fontStyle': 'italic' if 'syntax' in line.lower() or '语法' in line else 'normal'
            }
        ) for line in SEARCH_EXAMPLES[language]
    ] 