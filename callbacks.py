import dash
from dash import Input, Output, State, html
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from flask_login import login_user, logout_user, UserMixin, current_user
import re
import json
from datetime import datetime, timedelta
import pandas as pd
from dateutil.relativedelta import relativedelta
from dash import dcc
from collections import Counter
import nltk
from nltk.corpus import stopwords
from urllib.parse import parse_qs
from typing import List, Dict, Optional, Tuple # Added types

from config import TRANSLATIONS, AXIS_CATEGORY_NAMES, VALID_USERNAME, VALID_PASSWORD, color_mapping, color_to_prefix, type_colors
# Removed get_review_format import as it's handled differently
# Make sure get_review_format is removed or adapted if it was used elsewhere
from data import get_cached_dict, normalize_search_query, get_bar_chart_data, get_plot_data, get_review_date_range, get_category_time_series, update_raw_dict_map
from utils import ratio_to_rgb, get_size_legend_translations, get_hover_translations, get_proportional_size, get_search_examples_html

# Initialize NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.update(['said', 'would', 'also', 'could', 'one', 'like', 'still', 'even', 'get', 'use', 'used'])

def extract_text_from_review(review_info: Dict) -> str:
    """
    Extract plain text (title + content) from a review_info dictionary.
    
    Args:
        review_info: Dictionary from extract_review_info
        
    Returns:
        Plain text content of the review (title + content)
    """
    if not review_info:
        return ""
    # Combine title and content for full text
    title = review_info.get('review_title', '')
    content = review_info.get('review_content', '')
    # Return raw title + content, stripping extra spaces
    # The full_review_text already has <b> tags, we need the content part
    return f"{title} {content}".strip()

def extract_highlight_text(review_info: Dict, axis='x') -> str:
    """
    Extract text from highlights (detail or reason) for a specific axis.
    
    Args:
        review_info: Dictionary from extract_review_info
        axis: 'x' (detail) or 'y' (reason) to specify which highlight text to extract
        
    Returns:
        The highlight text (detail or reason)
    """
    if not review_info:
        return ""
    if axis == 'x':
        return review_info.get('highlight_detail_text', '')
    elif axis == 'y':
        return review_info.get('highlight_reason_text', '') or '' # Return empty string if None
    return ""

def get_all_ngrams_from_review(review_info: Dict) -> List[str]:
    """Extract all ngram texts from a review_info dictionary."""
    if not review_info:
        return []
    ngrams = []
    ngrams.extend([ngram['text'] for ngram in review_info.get('highlight_detail_ngrams', [])])
    ngrams.extend([ngram['text'] for ngram in review_info.get('highlight_reason_ngrams', [])])
    ngrams.extend([ngram['text'] for ngram in review_info.get('review_ngrams', [])])
    return [ngram.lower() for ngram in ngrams]

def get_frequent_words(
    review_dicts: List[Dict],
    axis: Optional[str] = None,
    num_words: int = 15
) -> List[Tuple[str, int]]:
    """
    Extract the most frequent words/ngrams from a list of review dictionaries.
    
    Args:
        review_dicts: List of review dictionaries from extract_review_info
        axis: 'x' (detail) or 'y' (reason) to extract highlight text, or None for full text
        num_words: Number of top words/ngrams to return
        
    Returns:
        List of (word, count) tuples for the most frequent words/ngrams
    """
    all_ngrams = []
    all_words_text = []

    # Remove None entries
    filtered_reviews = [r for r in review_dicts if r]

    # Extract text and ngrams from the filtered reviews
    for review_info in filtered_reviews:
        # Extract ngrams based on axis parameter
        if axis == 'x':
            # Only get ngrams from highlight_detail_ngrams for X-axis
            detail_ngrams = [ngram['text'].lower() for ngram in review_info.get('highlight_detail_ngrams', [])]
            all_ngrams.extend(detail_ngrams)
        elif axis == 'y':
            # Only get ngrams from highlight_reason_ngrams for Y-axis
            reason_ngrams = [ngram['text'].lower() for ngram in review_info.get('highlight_reason_ngrams', [])]
            all_ngrams.extend(reason_ngrams)
        else:
            # For no specific axis, get all ngrams
            all_ngrams.extend(get_all_ngrams_from_review(review_info))

        # Extract relevant text (highlight or full)
        if axis:
            text = extract_highlight_text(review_info, axis)
        else:
            text = extract_text_from_review(review_info)
        all_words_text.append(text.lower())
    
    # Count ngrams
    ngram_counts = Counter(all_ngrams)
    # Return the most common words/ngrams with minimum occurrences (e.g., >= 2)
    min_occurrences = 2
    most_common = [(word, count) for word, count in ngram_counts.most_common(num_words * 2) if count >= min_occurrences]

    return most_common[:num_words]

def create_word_frequency_display(word_counts, language, word_type='common', on_click_id=None, selected_words=None, title_word=None):
    """
    Create a clickable display of word frequencies.
    
    Args:
        word_counts: List of (word, count) tuples
        language: Current language setting
        word_type: Type of words ('positive', 'negative', or 'common')
        on_click_id: ID for the div to enable click events
        selected_words: List of currently selected words
        title_word: Category name to include in title
        
    Returns:
        HTML div containing the word cloud display
    """
    if not word_counts:
        return html.Div()
    
    selected_words = selected_words or []
    
    # Determine color based on word type
    if word_type == 'positive':
        base_color = '#E6F3FF'  # Light blue for positive
        title = TRANSLATIONS[language]['frequent_positive_words']
        selected_color = '#6BB5FF'  # Even darker blue for selected
        title_style = {'marginBottom': '5px'}
        category_style = {}
    elif word_type == 'negative':
        base_color = '#FFE6E6'  # Light red for negative
        title = TRANSLATIONS[language]['frequent_negative_words']
        selected_color = '#FFB5B5'  # Even darker red for selected
        title_style = {'marginBottom': '5px'}
        category_style = {}
    else:
        if on_click_id == 'x-axis-words-container':
            base_color = '#FFFDE6'  # Light yellow for X-axis
            title = TRANSLATIONS[language]['frequent_words']
            selected_color = '#FFEC51'  # Even darker yellow for selected
            title_style = {'marginBottom': '5px'}
            category_style = {
                'border': f'5px solid {color_mapping["x"]}',  # Purple border for X-axis
                'padding': '5px',
                'display': 'inline-block',
                'marginLeft': '10px'
            }
        elif on_click_id == 'y-axis-words-container':
            base_color = '#E6FFE6'  # Light green for Y-axis
            title = TRANSLATIONS[language]['frequent_words']
            selected_color = '#51FF51'  # Even darker green for selected
            title_style = {'marginBottom': '5px'}
            category_style = {
                'border': f'5px solid {color_mapping["y"]}',  # Green border for Y-axis
                'padding': '5px',
                'display': 'inline-block',
                'marginLeft': '10px'
            }
        else:
            base_color = '#F0F0F0'  # Light gray for common
            title = TRANSLATIONS[language]['frequent_words']
            selected_color = '#C0C0C0'  # Even darker gray for selected
            title_style = {'marginBottom': '5px'}
            category_style = {}
    
    # Create clickable word buttons
    word_buttons = []
    for i, (word, count) in enumerate(word_counts):
        # Calculate font size based on frequency (between 9 and 19)
        # Find the max count to normalize
        max_count = max([c for _, c in word_counts])
        font_size = 13 + (count / max_count) * 6
        
        # Determine if this word is selected
        is_selected = word in selected_words
        
        # Set style based on selection state
        button_style = {
            'margin': '2px',
            'padding': '2px 4px',
            'fontSize': f'{font_size}px',
            'backgroundColor': selected_color if is_selected else base_color,
            'color': '#000' if is_selected else '#333',
            'border': '1px solid #aaa' if is_selected else '1px solid #ddd',
            'borderRadius': '8px',
            'cursor': 'pointer',
            'fontWeight': 'bold' if is_selected else 'normal',
            'boxShadow': '0px 2px 3px rgba(0,0,0,0.1)' if is_selected else 'none',
            'transition': 'all 0.2s ease',
            'display': 'inline-block'
        }
        
        # Use direct button without wrapper div
        word_buttons.append(
            html.Button(
                f"{word} ({count})",
                id={'type': 'word-button', 'index': word},
                style=button_style,
                n_clicks=0,
                title="Click to filter reviews containing this word"  # HTML tooltip
            )
        )
    
    return html.Div([
        html.H4([
            html.Span(f"{title_word} ", style=category_style) if title_word else None,
            html.Span(title, style=title_style)
        ]),
        html.Div(
            word_buttons,
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'gap': '5px',
                'padding': '5px',
                'border': '1px solid #ddd',
                'borderRadius': '5px',
                'marginBottom': '10px',
                'backgroundColor': '#fff'
            },
            id=on_click_id
        )
    ])

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

# User class for authentication
class User(UserMixin):
    def __init__(self, username):
        self.id = username
        
    def is_authenticated(self):
        return True
        
    def is_active(self):
        return True
        
    def is_anonymous(self):
        return False
        
    def get_id(self):
        return self.id

def register_callbacks(app):
    """Register all callbacks with the app."""
    
    @app.callback(
        [Output('url', 'pathname'),
         Output('url', 'search'),
         Output('login-error', 'children')],
        [Input('login-button', 'n_clicks')],
        [State('username-input', 'value'),
         State('password-input', 'value'),
         State('app-language-state', 'children'),
         State('category-state', 'children')],
        prevent_initial_call=True
    )
    def login_callback(n_clicks, username, password, language, category):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
            
        # Check if username and password are None or empty
        if not username or not password:
            return '/login', f'?lang={language}&category={category}', TRANSLATIONS[language]['invalid_credentials']
            
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            user = User(username)
            login_success = login_user(user)
            
            # Return to home page with language and category parameters preserved
            return '/', f'?lang={language}&category={category}', ''
        
        # Return to login page with error and parameters
        return '/login', f'?lang={language}&category={category}', TRANSLATIONS[language]['invalid_credentials']

    @app.callback(
        [Output('url', 'pathname', allow_duplicate=True),
         Output('url', 'search', allow_duplicate=True)],
        [Input('logout-button', 'n_clicks')],
        [State('url', 'pathname'),
         State('app-language-state', 'children'),
         State('category-state', 'children')],
        prevent_initial_call=True
    )
    def logout_callback(n_clicks, pathname, language, category):
        if n_clicks:
            logout_user()
            # Redirect to login page with parameters
            return '/login', f'?lang={language}&category={category}'
        # Default case - stay on current page
        return pathname, f'?lang={language}&category={category}'

    # Language-dependent callbacks
    @app.callback(
        [Output('plot-type-label', 'children'),
        Output('plot-type', 'options')],
        [Input('app-language-state', 'children')]
    )
    def update_plot_type_labels(language):
        options = [
            {'label': TRANSLATIONS[language]['bar_chart'], 'value': 'bar_chart'},
            {'label': TRANSLATIONS[language]['use_vs_attr_perf'], 'value': 'use_attr_perf'},
            {'label': TRANSLATIONS[language]['perf_vs_attr'], 'value': 'perf_attr'}
        ]
        return TRANSLATIONS[language]['plot_type'], options

    @app.callback(
        [Output('bar-chart-controls', 'style'),
         Output('matrix-view-controls', 'style')],
        [Input('plot-type', 'value')]
    )
    def toggle_controls_visibility(plot_type):
        if plot_type == 'bar_chart':
            return (
                {'display': 'block'}, # bar chart controls
                {'display': 'none'}   # matrix view controls
            )
        else:
            return (
                {'display': 'none'},  # bar chart controls
                {'display': 'block'}  # matrix view controls
            )

    @app.callback(
        [Output('bar-category-label', 'children'),
         Output('bar-zoom-label', 'children'),
         Output('bar-count-label', 'children'),
         Output('bar-category-checklist', 'options')],
        [Input('app-language-state', 'children')]
    )
    def update_bar_chart_labels(language):
        options = [
            {'label': TRANSLATIONS[language]['usage_category'], 'value': 'usage'},
            {'label': TRANSLATIONS[language]['attribute_category'], 'value': 'attribute'},
            {'label': TRANSLATIONS[language]['performance_category'], 'value': 'performance'}
        ]
        return (
            TRANSLATIONS[language]['bar_category_label'],
            TRANSLATIONS[language]['bar_zoom_label'],
            TRANSLATIONS[language]['bar_count_label'],
            options
        )

    @app.callback(
        Output('bar-count-slider', 'max'),
        [Input('bar-category-checklist', 'value'),
         Input('bar-zoom-dropdown', 'value'),
         Input('app-language-state', 'children'),
         Input('search-button', 'n_clicks'),
         Input('date-filter-slider', 'value')],
        [State('search-input', 'value'),
         State('date-filter-storage', 'children')]
    )
    def update_bar_slider_max(bar_categories, bar_zoom, language, n_clicks, date_slider_value, search_query, date_filter_storage):
        # Ensure bar_categories has valid values or use defaults
        if not bar_categories:
            bar_categories = ['usage', 'attribute', 'performance']
            
        # Parse date range from storage
        date_range = json.loads(date_filter_storage) if date_filter_storage else {"start_date": None, "end_date": None}
        start_date = date_range.get("start_date")
        end_date = date_range.get("end_date")
            
        # Get the categories and counts for the selected bar categories
        display_categories, original_categories, bar_counts, sentiment_ratios, review_data, colors, title_key = get_bar_chart_data(
            bar_categories, bar_zoom, language, search_query, start_date, end_date
        )
            
        # Return the count of categories or a minimum value if there are fewer categories
        return max(len(display_categories), 1)

    @app.callback(
        [Output('y-axis-label', 'children'),
         Output('x-axis-label', 'children'),
         Output('num-y-features-label', 'children'),
         Output('num-x-features-label', 'children')],
        [Input('app-language-state', 'children')]
    )
    def update_axis_labels(language):
        return (
            TRANSLATIONS[language]['y_axis_category'],
            TRANSLATIONS[language]['x_axis_category'],
            TRANSLATIONS[language]['num_y_features'],
            TRANSLATIONS[language]['num_x_features']
        )

    # Helper functions for modular callbacks
    def create_bar_chart(categories, counts, sentiment_ratios, colors, bar_zoom, language):
        """Create a bar chart figure based on the provided data."""
        # Create the bar chart figure
        fig = go.Figure()
        
        # Add bars
        for i, (category, count, sentiment, color) in enumerate(zip(categories, counts, sentiment_ratios, colors)):
            # Use the sentiment color for the bar fill
            fig.add_trace(go.Bar(
                x=[category],
                y=[count],
                name=category,
                marker=dict(
                    color=ratio_to_rgb(sentiment),  # Use sentiment color for bar
                    line=dict(
                        color='rgba(0,0,0,0.2)',  # Light gray border
                        width=1
                    )
                ),
                hovertemplate=(
                    f"{TRANSLATIONS[language]['hover_count']}: %{{y}}<br>" +
                    f"{TRANSLATIONS[language]['hover_satisfaction']}: {sentiment:.2f}"
                ),
                showlegend=False  # Hide individual bars from legend
            ))
        
        # Add an invisible satisfaction heatmap for color legend
        all_sentiments = np.array(sentiment_ratios).reshape(-1, 1)
        fig.add_trace(go.Heatmap(
            z=all_sentiments,
            colorscale=px.colors.diverging.RdBu,
            showscale=True,
            opacity=0,
            colorbar=dict(
                title=TRANSLATIONS[language]['satisfaction_level'],
                tickvals=[0, 0.5, 1],
                ticktext=[
                    TRANSLATIONS[language]['unsatisfied'],
                    TRANSLATIONS[language]['neutral'],
                    TRANSLATIONS[language]['satisfied']
                ],
                x=1.11,
                y=0.4
            )
        ))
        
        # Create custom x-axis ticktext with colored category labels
        ticktext = []
        
        # Check if we're zoomed into a parent category
        is_zoomed = bar_zoom and bar_zoom != ''
        
        for category in categories:
            # Determine the color based on the category prefix
            category_color = None
            for prefix, color_value in type_colors.items():
                if category.startswith(prefix):
                    category_color = color_value
                    break
            
            # Format display text with colored labels in HTML
            if category_color:
                ticktext.append(f'<span style="color:{category_color}; font-weight:bold">{category}</span>')
            else:
                ticktext.append(category)
        
        # Update layout for bar chart
        title_key = 'bar_chart_title'
        fig.update_layout(
            title=TRANSLATIONS[language][title_key],
            xaxis=dict(
                title=None,
                tickangle=20,
                tickmode='array',
                tickvals=list(range(len(categories))),
                ticktext=ticktext,
                domain=[0, 0.85]
            ),
            yaxis=dict(
                title=TRANSLATIONS[language]['hover_count'],
            ),
            showlegend=False,
            height=700,
            width=1200,
            margin=dict(
                l=80,
                r=150,
                t=100,
                b=180,
                autoexpand=True
            ),
        )
        
        # Adjust the chart height based on number of categories
        if len(categories) > 15:
            fig.update_layout(height=750)
        elif len(categories) > 10:
            fig.update_layout(height=720)
        
        # Update bar width based on number of categories
        if len(categories) <= 5:
            fig.update_layout(bargap=0.5)
        elif len(categories) <= 10:
            fig.update_layout(bargap=0.3)
        else:
            fig.update_layout(bargap=0.1)
        
        # Create a custom legend as an annotation
        legend_items = []
        for prefix, color in type_colors.items():
            if any(cat.startswith(prefix) for cat in categories):
                translated_label = TRANSLATIONS[language]['usage_category'] if prefix == 'U: ' else (
                    TRANSLATIONS[language]['attribute_category'] if prefix == 'A: ' else TRANSLATIONS[language]['performance_category']
                )
                if not translated_label.startswith(prefix):
                    translated_label = prefix + translated_label
                legend_items.append(f'<span style="color:{color}; font-weight:bold; margin-right:15px;">▣ {translated_label}</span>')
        
        if legend_items:
            legend_title = TRANSLATIONS[language].get('category_types', 'Category Types')
            legend_text = f'<b>{legend_title}:</b> ' + ' '.join(legend_items)
            
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=1.2,
                y=-0.22,
                text=legend_text,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.5)",
                borderwidth=1,
                borderpad=6,
                align="center"
            )
        
        return fig


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

    def create_heatmap(matrix, sentiment_matrix, review_matrix, x_text, y_text, language, plot_type):
        """Create a heatmap figure based on the provided data."""
        
        # Calculate dynamic dimensions
        # Height calculation
        base_height = 700
        min_height = 600
        additional_height_per_feature = 60
        dynamic_height = max(
            min_height,
            base_height + max(0, len(y_text) - 5) * additional_height_per_feature
        )
        
        # Width calculation
        base_width = 1000
        min_width = 1400
        additional_width_per_feature = 80
        max_width = 2400
        
        avg_label_length = sum(len(str(label)) for label in x_text) / len(x_text) if x_text else 0
        width_factor = max(1, avg_label_length / 15)
        
        dynamic_width = min(
            max_width,
            max(
                min_width,
                base_width + max(0, len(x_text) - 5) * additional_width_per_feature * width_factor
            )
        )
        
        # Calculate margins
        max_y_label_length = max(len(str(label)) for label in y_text) if y_text else 0
        left_margin = min(250, max(150, max_y_label_length * 9))
        right_margin = 400
        top_margin = 120
        bottom_margin = 180

        # Create the figure
        fig = go.Figure()
        
        # Handle empty matrix case
        if matrix.size == 0 or not np.any(matrix > 0):
            fig.add_annotation(
                x=0.5, y=0.5,
                text=TRANSLATIONS[language]['no_data_available'],
                font=dict(size=16),
                showarrow=False,
                xref="paper", yref="paper"
            )
            return fig
        
        # Add the satisfaction heatmap legend
        fig.add_trace(go.Heatmap(
            z=sentiment_matrix,
            x=x_text,
            y=y_text,
            colorscale=px.colors.diverging.RdBu,
            showscale=True,
            opacity=0,
            colorbar=dict(
                title=TRANSLATIONS[language]['satisfaction_level'],
                tickvals=[0, 0.5, 1],
                ticktext=[
                    TRANSLATIONS[language]['unsatisfied'],
                    TRANSLATIONS[language]['neutral'],
                    TRANSLATIONS[language]['satisfied']
                ],
                x=1.11,
                y=0.4
            )
        ))
        
        # Add the custom shapes
        max_mentions = np.max(matrix)
        min_mentions = np.min(matrix[matrix > 0])
        
        for i in range(len(y_text)):
            for j in range(len(x_text)):
                if matrix[i, j] > 0:
                    # Calculate square size based on the value
                    square_size = get_proportional_size(matrix[i, j], max_mentions, min_mentions)
                    # Create a square centered at (j,i)
                    fig.add_shape(
                        type="rect",
                        x0=j-square_size/2, y0=i-square_size/2, 
                        x1=j+square_size/2, y1=i+square_size/2,
                        fillcolor=ratio_to_rgb(sentiment_matrix[i, j]),
                        line_color="rgba(0,0,0,0)",
                    )
        
        # Add legend and examples
        size_legend_text = get_size_legend_translations(language)['explanation']
        
        # Find a good example box
        if np.any(matrix > 0):
            example_i, example_j = np.unravel_index(np.argmax(matrix), matrix.shape)
            example_sentiment = sentiment_matrix[example_i, example_j]
        else:
            example_sentiment = 0.5
        
        # Constants for example box and explanation
        EXAMPLE_BOX_LEFT = 1.08
        EXAMPLE_BOX_SIZE = 0.1  # Make a square
        EXAMPLE_BOX_RIGHT = EXAMPLE_BOX_LEFT + EXAMPLE_BOX_SIZE
        EXAMPLE_BOX_BOTTOM = 0.95
        EXAMPLE_BOX_TOP = EXAMPLE_BOX_BOTTOM + EXAMPLE_BOX_SIZE
        BRACKET_TOP = EXAMPLE_BOX_TOP + 0.02
        BRACKET_VERTICAL_LENGTH = 0.015
        EXPLANATION_X = EXAMPLE_BOX_LEFT + (EXAMPLE_BOX_RIGHT - EXAMPLE_BOX_LEFT) * 1.5
        EXPLANATION_Y = BRACKET_TOP + 0.03
        
        # Add explanation text
        fig.add_annotation(
            xref="paper", yref="paper", 
            x=EXPLANATION_X,
            y=EXPLANATION_Y,
            text=size_legend_text,
            showarrow=False,
            font=dict(size=13),
            align="right",
            width=220
        )
        
        # Add the example box (now a square)
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=EXAMPLE_BOX_LEFT, 
            x1=EXAMPLE_BOX_RIGHT,
            y0=EXAMPLE_BOX_BOTTOM, 
            y1=EXAMPLE_BOX_TOP,
            fillcolor=ratio_to_rgb(example_sentiment),
            line_color="rgba(0,0,0,0)",
        )
        
        # Add bracket at the top of the box
        fig.add_shape(
            type="line",
            xref="paper", yref="paper",
            x0=EXAMPLE_BOX_LEFT, 
            x1=EXAMPLE_BOX_RIGHT,
            y0=BRACKET_TOP, 
            y1=BRACKET_TOP,
            line=dict(color="black", width=1)
        )
        
        # Add vertical lines for bracket
        for x in [EXAMPLE_BOX_LEFT, EXAMPLE_BOX_RIGHT]:
            fig.add_shape(
                type="line",
                xref="paper", yref="paper",
                x0=x, x1=x,
                y0=BRACKET_TOP, 
                y1=BRACKET_TOP - BRACKET_VERTICAL_LENGTH,
                line=dict(color="black", width=1)
            )
        
        # Update axis titles with category types
        if plot_type == 'use_attr_perf':
            x_category_type = AXIS_CATEGORY_NAMES[language]['use']
            y_category_type = AXIS_CATEGORY_NAMES[language]['attr_perf']
            title_key = 'use_attr_perf_title'
        else:  # perf_attr
            x_category_type = AXIS_CATEGORY_NAMES[language]['perf']
            y_category_type = AXIS_CATEGORY_NAMES[language]['attr']
            title_key = 'perf_attr_title'
        
        fig.update_layout(
            title=TRANSLATIONS[language][title_key],
            xaxis=dict(
                tickangle=20,
                tickmode='array',
                tickvals=list(range(len(x_text))),
                ticktext=x_text,
                tickfont=dict(size=12),
                title=dict(
                    text=f"{TRANSLATIONS[language]['x_axis_percentage']} {x_category_type}",
                    font=dict(size=14)
                )
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(y_text))),
                ticktext=y_text,
                tickfont=dict(size=12),
                title=dict(
                    text=f"{y_category_type} {TRANSLATIONS[language]['y_axis_percentage']}",
                    font=dict(size=14)
                )
            ),
            font=dict(size=13),
            title_font=dict(size=18),
            width=dynamic_width,
            height=dynamic_height,
            margin=dict(
                l=left_margin,
                r=right_margin,
                t=top_margin,
                b=bottom_margin,
                autoexpand=True
            )
        )
        
        # Configure hover info - this is the important part that was missing
        hover_translations = get_hover_translations(language)
        hover_template = (
            f"{hover_translations['hover_x']}: %{{x}}<br>" +
            f"{hover_translations['hover_y']}: %{{y}}<br>" +
            f"{hover_translations['hover_count']}: %{{customdata[0]}}<br>" + # Use customdata index 0 for count
            f"{hover_translations['hover_satisfaction']}: %{{customdata[1]:.2f}}" # Use customdata index 1 for sentiment
        )
        
        # Combine matrix and sentiment_matrix for customdata
        custom_data_combined = np.stack((matrix, sentiment_matrix), axis=-1)
        
        fig.update_traces(
            hovertemplate=hover_template,
            customdata=custom_data_combined, # Pass combined data
            # text=matrix # Text is now part of customdata
        )
        
        return fig
        
    def process_bar_chart_data(bar_categories, bar_zoom_value, bar_count, language, search_query, start_date, end_date):
        """
        Process data for bar chart visualization and controls.
        
        Args:
            bar_categories: List of category types to include
            bar_zoom_value: Value of the selected zoom category
            bar_count: Maximum number of bars to display
            language: Current language setting
            search_query: Search query for filtering
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Tuple containing (figure, zoom_dropdown_options, new_zoom_value, updated_bar_count)
        """
        # Handle empty values
        if not bar_categories:
            bar_categories = ['usage', 'attribute', 'performance']
        if bar_zoom_value is None:
            bar_zoom_value = 'all'
        # Get bar chart data
        display_categories, original_categories, counts, sentiment_ratios, review_data, colors, title_key = get_bar_chart_data(
            bar_categories, bar_zoom_value, language, search_query, start_date, end_date
        )
        
        # Update bar count based on available categories
        bar_count = min(bar_count or 10, len(display_categories))
        
        # Apply bar count limit from bar_count slider for visualization
        if bar_count > 0 and bar_count < len(display_categories):
            display_categories = display_categories[:bar_count]
            original_categories = original_categories[:bar_count]
            display_counts = counts[:bar_count]
            display_sentiment_ratios = sentiment_ratios[:bar_count]
            display_colors = colors[:bar_count]
        else:
            display_counts = counts
            display_sentiment_ratios = sentiment_ratios
            display_colors = colors

        formatted_categories = [format_category_display(category, language) for category in display_categories]
        dropdown_options = get_options(language, bar_zoom_value, original_categories, formatted_categories)
        # Create the bar chart figure
        fig = create_bar_chart(formatted_categories, display_counts, display_sentiment_ratios, display_colors, bar_zoom_value, language)
        return fig, dropdown_options, bar_zoom_value, bar_count



    def process_matrix_data(plot_type, x_value, y_value, top_n_x, top_n_y, language, search_query, start_date, end_date):
        """
        Process data for matrix visualization and controls.
        
        Args:
            plot_type: Type of matrix plot ('use_attr_perf' or 'perf_attr')
            x_value: X-axis category selection
            y_value: Y-axis category selection
            top_n_x: Number of x-axis features to display
            top_n_y: Number of y-axis features to display
            language: Current language setting
            search_query: Search query for filtering
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Tuple containing (figure, x_options, y_options, max_x, max_y, updated_top_n_x, updated_top_n_y)
        """
        # Handle None values
        x_value = x_value or 'all'
        y_value = y_value or 'all'
        
        # Get data for the axes
        if plot_type == 'use_attr_perf':
            x_dict = get_cached_dict('use_sents', x_value, search_query, start_date, end_date)
            y_dict = get_cached_dict('attr_perf_ids', y_value, search_query, start_date, end_date)
        else:  # perf_attr
            x_dict = get_cached_dict('perf_sents', x_value, search_query, start_date, end_date)
            y_dict = get_cached_dict('attr_ids', y_value, search_query, start_date, end_date)
        
        # Update slider maximums based on available features
        max_x = max(1, len(x_dict))
        max_y = max(1, len(y_dict))
        
        # Ensure top_n values don't exceed available features
        top_n_x = min(top_n_x, max_x)
        top_n_y = min(top_n_y, max_y)
        
        # Get plot data for both visualization and dropdown options
        # This now returns review_dict_matrix instead of review_matrix
        matrix, sentiment_matrix, review_dict_matrix, x_text, x_percentages, y_text, y_percentages, title_key = get_plot_data(
            plot_type, x_value, y_value, max_x, max_y, language, search_query, start_date, end_date
        )
        
        # Reorder x-axis by frequency for visualization
        if len(matrix) > 0 and len(matrix[0]) > 0:
            x_sorted_indices = np.argsort(-x_percentages)[:top_n_x]
            
            # Create sorted versions of data for visualization
            viz_matrix = matrix[:, x_sorted_indices]
            viz_sentiment_matrix = sentiment_matrix[:, x_sorted_indices]
            # viz_review_matrix = review_matrix[:, x_sorted_indices] # Old
            # Correct slicing for review_dict_matrix (numpy array of lists/objects)
            viz_review_dict_matrix = review_dict_matrix[:, x_sorted_indices]

            viz_x_text = [x_text[i] for i in x_sorted_indices]

            y_sorted_indices = np.argsort(-y_percentages)[:top_n_y]
            viz_matrix = viz_matrix[y_sorted_indices, :]
            viz_sentiment_matrix = viz_sentiment_matrix[y_sorted_indices, :]
            # viz_review_matrix = viz_review_matrix[y_sorted_indices, :] # Old
            viz_review_dict_matrix = viz_review_dict_matrix[y_sorted_indices, :]
            viz_y_text = [y_text[i] for i in y_sorted_indices]
        else:
            # Handle empty matrix case
            viz_matrix = matrix
            viz_sentiment_matrix = sentiment_matrix
            # viz_review_matrix = review_matrix # Old
            viz_review_dict_matrix = review_dict_matrix
            viz_x_text = x_text
            viz_y_text = y_text

        # Format display labels - remove all parentheses content for all categories
        formatted_x_labels = [format_category_display(label, language) for label in viz_x_text]
        formatted_y_labels = [format_category_display(label, language) for label in viz_y_text]
        x_options = get_options(language, x_value, viz_x_text, formatted_x_labels)
        y_options = get_options(language, y_value, viz_y_text, formatted_y_labels)

        # Create clean text without percentages for lookups
        if x_value == 'all':
            x_axis_text = formatted_x_labels
        else:
            # For zoomed categories, show only the child categories
            x_axis_text = []
            for label in formatted_x_labels:
                if '|' in label:
                    parts = label.split('|')
                    if len(parts) > 1:
                        x_axis_text.append('|'.join(parts[1:]))
                    else:
                        x_axis_text.append(parts[0])
                else:
                    x_axis_text.append(label)

        if y_value == 'all':
            y_axis_text = formatted_y_labels
        else:
            # For zoomed categories, show only the child categories
            y_axis_text = []
            for label in formatted_y_labels:
                if '|' in label:
                    parts = label.split('|')
                    if len(parts) > 1:
                        y_axis_text.append('|'.join(parts[1:]))
                    else:
                        y_axis_text.append(parts[0])
                else:
                    y_axis_text.append(label)

        # Create the heatmap figure
        fig = create_heatmap(
            viz_matrix, viz_sentiment_matrix, viz_review_dict_matrix, # Pass dict matrix
            x_axis_text, y_axis_text,
            language, plot_type
        )
        
        return fig, x_options, y_options, max_x, max_y, top_n_x, top_n_y

    def handle_date_filter(trigger_id, date_slider_value, date_filter_storage):
        """
        Handle date filter initialization and updates.
        
        Args:
            trigger_id: ID of the component that triggered the callback
            date_slider_value: Current value of the date slider
            date_filter_storage: Current date filter storage JSON
            
        Returns:
            Tuple containing (updated_date_storage, slider_min, slider_max, slider_value, slider_marks)
        """
        # Default slider settings for no-update cases
        slider_min = dash.no_update
        slider_max = dash.no_update
        slider_value = dash.no_update
        slider_marks = dash.no_update
        
        # Initialize storage if empty or on first load
        storage_data = {}
        if not date_filter_storage:
            # Get min and max dates from all reviews
            min_date_str, max_date_str = get_review_date_range()
            
            if min_date_str is None or max_date_str is None:
                # No dates found, use empty values
                return json.dumps({"start_date": None, "end_date": None}), 0, 1, [0, 1], {}
            
            # Convert to datetime objects
            min_date = datetime.strptime(min_date_str, '%Y-%m-%d')
            max_date = datetime.strptime(max_date_str, '%Y-%m-%d')
            
            # Set up slider values
            slider_min = 0
            slider_max = (max_date - min_date).days
            slider_value = [slider_min, slider_max]
            
            # Create slider marks
            slider_marks = {}
            
            # Determine appropriate frequency based on date range
            date_range_days = (max_date - min_date).days
            
            # Use quarterly marks for ranges under 3 years, 6-monthly for 3-10 years, yearly for longer
            if date_range_days <= 365 * 3:
                # Quarterly marks
                mark_dates = pd.date_range(start=min_date, end=max_date, freq='QS')
            elif date_range_days <= 365 * 10:
                # 6-monthly marks
                # Create a custom range starting from min_date and adding 6 months each time
                mark_dates = []
                current_date = min_date
                while current_date <= max_date:
                    mark_dates.append(current_date)
                    current_date = current_date + relativedelta(months=6)
                mark_dates = pd.DatetimeIndex(mark_dates)
            else:
                # Yearly marks
                mark_dates = pd.date_range(start=min_date, end=max_date, freq='YS')
            
            # Add marks for each date in the range
            for date in mark_dates:
                mark_value = (date - min_date).days
                
                # Format date as YYYY-MM
                date_label = date.strftime('%Y-%m')
                
                # Style the marks
                slider_marks[mark_value] = {
                    'label': date_label,
                    'style': {'transform': 'rotate(45deg)', 'white-space': 'nowrap'}
                }
            
            # Always include min and max dates with special styling
            slider_marks[slider_min] = {
                'label': min_date.strftime('%Y-%m'),
                'style': {'color': '#2196F3', 'font-weight': 'bold', 'transform': 'rotate(45deg)'}
            }
            slider_marks[slider_max] = {
                'label': max_date.strftime('%Y-%m'),
                'style': {'color': '#2196F3', 'font-weight': 'bold', 'transform': 'rotate(45deg)'}
            }
            
            # Initialize storage data
            storage_data = {
                "min_date": min_date_str,
                "max_date": max_date_str,
                "start_date": min_date_str,
                "end_date": max_date_str
            }
        # Update storage based on slider change
        elif trigger_id == 'date-filter-slider.value':
            try:
                storage_data = json.loads(date_filter_storage)
                
                # Ensure we have min_date
                if "min_date" not in storage_data:
                    min_date_str, max_date_str = get_review_date_range()
                    if min_date_str is None:
                        return json.dumps({"start_date": None, "end_date": None}), slider_min, slider_max, slider_value, slider_marks
                    storage_data["min_date"] = min_date_str
                    storage_data["max_date"] = max_date_str
                
                # Calculate new dates based on slider values
                min_date = datetime.strptime(storage_data["min_date"], '%Y-%m-%d')
                start_date = (min_date + timedelta(days=date_slider_value[0])).strftime('%Y-%m-%d')
                end_date = (min_date + timedelta(days=date_slider_value[1])).strftime('%Y-%m-%d')
                
                # Update only what changed
                storage_data["start_date"] = start_date
                storage_data["end_date"] = end_date
            except Exception as e:
                print(f"Error updating date filter: {str(e)}")
                return json.dumps({"start_date": None, "end_date": None}), slider_min, slider_max, slider_value, slider_marks
        else:
            # For other triggers, just parse existing storage
            storage_data = json.loads(date_filter_storage) if date_filter_storage else {"start_date": None, "end_date": None}
            
        # Return the updated storage and slider settings
        return json.dumps(storage_data), slider_min, slider_max, slider_value, slider_marks

    # Callback for combined graph and UI control updates
    @app.callback(
        [Output('main-figure', 'figure'),
         Output('reviews-content', 'children'),
         Output('x-axis-dropdown', 'options'),
         Output('y-axis-dropdown', 'options'),
         Output('x-axis-dropdown', 'value'),
         Output('y-axis-dropdown', 'value'),
         Output('x-features-slider', 'max'),
         Output('y-features-slider', 'max'),
         Output('x-features-slider', 'value'),
         Output('y-features-slider', 'value'),
         Output('bar-count-slider', 'value'),
         Output('bar-zoom-dropdown', 'options'),
         Output('bar-zoom-dropdown', 'value'),
         Output('date-filter-storage', 'children'),
         Output('date-filter-slider', 'min'),
         Output('date-filter-slider', 'max'),
         Output('date-filter-slider', 'value'),
         Output('date-filter-slider', 'marks'),
         Output('total-reviews-count', 'data')],
        [Input('plot-type', 'value'),
         Input('x-axis-dropdown', 'value'),
         Input('y-axis-dropdown', 'value'),
         Input('x-features-slider', 'value'),
         Input('y-features-slider', 'value'),
         Input('app-language-state', 'children'),
         Input('search-button', 'n_clicks'),
         Input('bar-category-checklist', 'value'),
         Input('bar-zoom-dropdown', 'value'),
         Input('bar-count-slider', 'value'),
         Input('date-filter-slider', 'value'),
         Input('url', 'pathname'),
         Input('url', 'search')],  # Add URL search parameter as input
        [State('search-input', 'value'),
         State('date-filter-storage', 'children')]
    )
    def update_visualization_and_controls(plot_type, x_value, y_value, top_n_x, top_n_y, language, n_clicks, 
                   bar_categories, bar_zoom_value, bar_count, date_slider_value, pathname, search, search_query, date_filter_storage):
        """
        Unified callback to update both the graph visualization and UI controls.
        This eliminates redundant calculations by using helper functions for specific plot types.
        """
        # Get category from URL search parameters
        try:
            query_params = parse_qs(search.lstrip('?'))
            category = query_params.get('category', ['Cables'])[0]
        except:
            category = 'Cables'
        
        # Update data for the current category
        update_raw_dict_map(category)
        
        # Check which input triggered the callback
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
        
        # Handle date filter logic
        updated_date_storage, slider_min, slider_max, slider_value, slider_marks = handle_date_filter(
            trigger_id, date_slider_value, date_filter_storage
        )
        
        # Parse date range from storage (common to both operations)
        date_range = json.loads(updated_date_storage) if updated_date_storage else {"start_date": None, "end_date": None}
        start_date = date_range.get("start_date")
        end_date = date_range.get("end_date")
        
        # Initialize reviews content
        reviews_content = [] if trigger_id == 'search-button.n_clicks' else dash.no_update
        
        # Clear reviews if the plot type changes
        if trigger_id == 'plot-type.value':
            reviews_content = []

        # Handle None values with defaults
        search_query = search_query if search_query else ''
        x_value = x_value or 'all'
        y_value = y_value or 'all'
        top_n_x = int(top_n_x) if top_n_x is not None else 10
        top_n_y = int(top_n_y) if top_n_y is not None else 10
        
        # Reset selections if search button was clicked
        if trigger_id == 'search-button.n_clicks':
            x_value = 'all'
            y_value = 'all'
            bar_zoom_value = 'all' # Reset bar zoom too
        
        # Calculate total reviews count if date changes, search changes, or on initial load
        review_count_text = dash.no_update
        should_update_count = (
            trigger_id == 'date-filter-slider.value' or 
            trigger_id == 'search-button.n_clicks' or 
            trigger_id == 'url.pathname' or
            not date_filter_storage  # Initial load
        )
        
        if should_update_count:
            # Get all categories for counting
            categories = ['usage', 'attribute', 'performance']
            total_reviews = 0
            unique_review_ids = set()
            
            # Count unique reviews across all categories
            for category_type in categories:
                dict_type = {
                    'usage': 'use_sents',
                    'attribute': 'attr_sents',
                    'performance': 'perf_sents'
                }[category_type]
                
                # Get filtered dictionary
                filtered_dict = get_cached_dict(dict_type, 'all', search_query, start_date, end_date)
                
                # Count unique reviews
                for path, sents_dict in filtered_dict.items():
                    for sent, val in sents_dict.items():
                        if isinstance(val, dict):
                            for rid, reviews in val.items():
                                if reviews:
                                    unique_review_ids.update([rid])
                        elif isinstance(val, list) and val:
                            # This is a simplified case, might need adjustment based on actual data structure
                            total_reviews += len(val)
            
            # Add unique reviews to total
            total_reviews = len(unique_review_ids) # Use unique count as primary
            
            # Format the output message based on language
            if search_query:
                review_count_text = f"{TRANSLATIONS[language].get('total_reviews_filtered', 'Total reviews matching filter')}: {total_reviews}"
            else:
                review_count_text = f"{TRANSLATIONS[language].get('total_reviews', 'Total reviews')}: {total_reviews}"
        elif trigger_id == 'app-language-state.value':
            # Just update the text without recounting when language changes
            # Get the current text and extract the number
            current_count = dash.callback_context.states.get('total-reviews-count.data')
            
            # If we have a current count, reuse it with the new language
            if current_count:
                try:
                    total_reviews = int(current_count)
                    if search_query:
                        review_count_text = f"{TRANSLATIONS[language].get('total_reviews_filtered', 'Total reviews matching filter')}: {total_reviews}"
                    else:
                        review_count_text = f"{TRANSLATIONS[language].get('total_reviews', 'Total reviews')}: {total_reviews}"
                except:
                    # If parsing fails, leave as no_update
                    pass
        
        # Process data based on plot type
        if plot_type == 'bar_chart':
            # Check if bar_zoom triggered the callback and reset bar_count if it did
            if trigger_id == 'bar-zoom-dropdown.value':
                bar_count = 10  # Reset to default value
                
            # Process bar chart data using helper function
            fig, dropdown_options, new_zoom_value, bar_count = process_bar_chart_data(
                bar_categories, bar_zoom_value, bar_count, language, search_query, start_date, end_date
            )
            
            # Return with default values for matrix-specific outputs
            return (
                fig,                    # figure
                reviews_content,        # reviews-content
                [],                     # x-axis-dropdown options
                [],                     # y-axis-dropdown options
                x_value,                # x-axis-dropdown value
                y_value,                # y-axis-dropdown value
                10,                     # x-features-slider max
                10,                     # y-features-slider max
                top_n_x,                # x-features-slider value
                top_n_y,                # y-features-slider value
                bar_count,              # bar-count-slider value
                dropdown_options,       # bar-zoom-dropdown options
                new_zoom_value,         # bar-zoom-dropdown value
                updated_date_storage,   # date-filter-storage
                slider_min,             # date-filter-slider min
                slider_max,             # date-filter-slider max
                slider_value,           # date-filter-slider value
                slider_marks,           # date-filter-slider marks
                total_reviews if should_update_count else dash.no_update  # total-reviews-count: store the count value only
            )
        else:
            # Check if axis dropdowns triggered the callback and reset corresponding top_n values
            if trigger_id == 'x-axis-dropdown.value':
                top_n_x = 10  # Reset to default value
            if trigger_id == 'y-axis-dropdown.value':
                top_n_y = 10  # Reset to default value
                
            # Process matrix data using helper function
            fig, x_options, y_options, max_x, max_y, top_n_x, top_n_y = process_matrix_data(
                plot_type, x_value, y_value, top_n_x, top_n_y, language, search_query, start_date, end_date
            )
            
            # Return values for matrix view (with no change to bar chart controls)
            return (
                fig,                    # figure
                reviews_content,        # reviews-content
                x_options,              # x-axis-dropdown options
                y_options,              # y-axis-dropdown options
                x_value,                # x-axis-dropdown value
                y_value,                # y-axis-dropdown value
                max_x,                  # x-features-slider max
                max_y,                  # y-features-slider max
                top_n_x,                # x-features-slider value
                top_n_y,                # y-features-slider value
                dash.no_update,         # bar-count-slider value
                dash.no_update,         # bar-zoom-dropdown options
                dash.no_update,         # bar-zoom-dropdown value
                updated_date_storage,   # date-filter-storage
                slider_min,             # date-filter-slider min
                slider_max,             # date-filter-slider max
                slider_value,           # date-filter-slider value
                slider_marks,           # date-filter-slider marks
                total_reviews if should_update_count else dash.no_update  # total-reviews-count: store the count value only
            )

    # Helper functions to reduce duplication
    def categorize_reviews(review_dicts: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Sort review dictionaries into positive, negative, and neutral based on star rating."""
        positive_reviews = []
        negative_reviews = []
        neutral_reviews = []

        for review_info in review_dicts:
            if not review_info: continue
            stars = review_info.get('star_rating', '').count('★')
            if stars >= 4:
                positive_reviews.append(review_info)
            elif stars <= 2:
                negative_reviews.append(review_info)
            else:
                neutral_reviews.append(review_info)

        return positive_reviews, negative_reviews, neutral_reviews

    def filter_reviews_by_sentiment(all_review_dicts, positive_review_dicts, negative_review_dicts, sentiment_filter):
        """Filter reviews based on sentiment selection"""
        if sentiment_filter == 'show_positive':
            return positive_review_dicts # Corrected: Use the parameter name
        elif sentiment_filter == 'show_negative':
            return negative_review_dicts # Corrected: Use the parameter name
        else:  # 'show_all'
            return all_review_dicts # Corrected: Use the parameter name

    def format_review_html(review_info: Dict, selected_words: Optional[List[str]] = None) -> str:
        """Creates the HTML for a single review with potential ngram highlighting."""
        if not review_info:
            return ""

        # Basic info
        product_id = review_info.get('product_id', '')
        rating = review_info.get('star_rating', '')
        date = review_info.get('date', '')
        # title = review_info.get('review_title', '') # Title is already in full_review_text with <b>
        # content = review_info.get('review_content', '') # Content is also part of full_review_text
        full_review_text_raw = review_info.get('full_review_text', '') # Includes <b> tags for title
        review_ngrams = review_info.get('review_ngrams', [])
        highlight_detail = review_info.get('highlight_detail_text', '')
        highlight_detail_ngrams = review_info.get('highlight_detail_ngrams', [])
        highlight_reason = review_info.get('highlight_reason_text', '')
        highlight_reason_ngrams = review_info.get('highlight_reason_ngrams', [])

        # --- Helper for Ngram Highlighting ---
        def apply_ngram_highlights(text: str, ngrams: List[Dict], selected_words_lower: List[str]) -> str:
            if not selected_words_lower or not ngrams:
                return text

            # Filter and sort ngrams matching selected words
            matching_ngrams = sorted(
                [ngram for ngram in ngrams if ngram['text'].lower() in selected_words_lower],
                key=lambda x: x['start'] # Sort by start position
            )

            if not matching_ngrams:
                return text

            highlighted_text = ""
            current_pos = 0
            highlight_tag_open = '<span style="background-color: #FFFF00; font-weight: bold;">' # Bright yellow
            highlight_tag_close = '</span>'

            for ngram in matching_ngrams:
                start, end = ngram['start'], ngram['end']
                # Add text before the current ngram
                if start > current_pos:
                    highlighted_text += text[current_pos:start]
                # Add the highlighted ngram itself, checking bounds
                if start < end <= len(text):
                    highlighted_text += highlight_tag_open + text[start:end] + highlight_tag_close
                # Update current position, handling potential overlaps (take the furthest end)
                current_pos = max(current_pos, end)

            # Add any remaining text after the last ngram
            if current_pos < len(text):
                highlighted_text += text[current_pos:]

            return highlighted_text

        # Prepare selected words list (lowercase)
        selected_words_lower = [w.lower() for w in selected_words] if selected_words else []

        # --- Highlight Ngrams in Full Review Text ---
        highlighted_full_review = apply_ngram_highlights(full_review_text_raw, review_ngrams, selected_words_lower)

        # --- Create Highlight Spans (Detail/Reason) ---
        def create_highlight_span(text, ngrams, axis_color, sentiment_color, selected_words_lower):
            highlighted_inner_text = apply_ngram_highlights(text, ngrams, selected_words_lower)
            return f'<span style="background-color: {sentiment_color}; border: 4px solid {axis_color}; color: black; padding: 2px;">{highlighted_inner_text}</span>'

        # Determine sentiment color based on rating
        stars_count = rating.count('★')
        sent_color = color_mapping.get('+') if stars_count >= 4 else (color_mapping.get('-') if stars_count <= 2 else color_mapping.get('?'))

        highlight_spans = []
        if highlight_detail:
            highlight_spans.append(create_highlight_span(highlight_detail, highlight_detail_ngrams, color_mapping['x'], sent_color, selected_words_lower))
        if highlight_reason:
            highlight_spans.append(create_highlight_span(highlight_reason, highlight_reason_ngrams, color_mapping['y'], sent_color, selected_words_lower))

        # --- Assemble Final HTML ---
        # Ensure spaces between components
        parts = [part for part in [
            ' '.join(highlight_spans),
            product_id,
            rating,
            f"[{date}]",
            highlighted_full_review # This already contains title + content
        ] if part] # Filter out empty parts

        return " ".join(parts)


    def create_review_display(filtered_review_dicts: List[Dict], language, plot_type='matrix', x_category=None, y_category=None, selected_words=None):
        """Create HTML content for reviews display using review dictionaries."""
        if not filtered_review_dicts:
            return [html.P(TRANSLATIONS[language]['no_reviews'])]
            
        # Initialize containers for word frequency display
        word_freq_components = []
        
        # Create word frequency analysis if there are enough reviews
        if len(filtered_review_dicts) >= 5: # Lower threshold for showing words
            # Get frequent words from all reviews
            all_reviews_words = get_frequent_words(filtered_review_dicts, num_words=25)
            
            # For matrix view, get X and Y axis highlights
            if plot_type == 'matrix':
                x_highlight_words = get_frequent_words(filtered_review_dicts, axis='x', num_words=25)
                y_highlight_words = get_frequent_words(filtered_review_dicts, axis='y', num_words=25)
                
                # Create word cloud displays for X and Y axis
                if x_highlight_words:
                    word_freq_components.append(
                        create_word_frequency_display(
                            x_highlight_words, 
                            language, 
                            'common',
                            on_click_id='x-axis-words-container',
                            selected_words=selected_words,
                            title_word=format_category_display(x_category, language) # Format title
                        )
                    )
                
                if y_highlight_words:
                    word_freq_components.append(
                        create_word_frequency_display(
                            y_highlight_words, 
                            language, 
                            'common',
                            on_click_id='y-axis-words-container',
                            selected_words=selected_words,
                            title_word=format_category_display(y_category, language) # Format title
                        )
                    )
            else:
                # For bar chart, get only X axis highlights
                # Use the detail highlight (axis='x')
                x_highlight_words = get_frequent_words(filtered_review_dicts, axis='x', num_words=25)
                if x_highlight_words:
                    word_freq_components.append(
                        create_word_frequency_display(
                            x_highlight_words, 
                            language, 
                            'common',
                            on_click_id='x-axis-words-container',
                            selected_words=selected_words,
                            title_word=format_category_display(x_category, language) # Format title
                        )
                    )
            
            # Add frequent words from all reviews
            if all_reviews_words:
                word_freq_components.append(
                    create_word_frequency_display(
                        all_reviews_words, 
                        language, 
                        'common',
                        on_click_id='all-reviews-words-container',
                        selected_words=selected_words,
                        title_word=TRANSLATIONS[language]['all_reviews'] # Title for all reviews
                    )
                )
                
            # Add buttons for selecting and deselecting all words
            word_freq_components.append(html.Div([
                html.Button(
                    TRANSLATIONS[language]['clear_word_selection'],
                    id='clear-word-selection-button',
                    style={
                        'padding': '6px 12px',
                        'backgroundColor': '#a0a0c0',
                        'border': '1px solid #ddd',
                        'borderRadius': '4px',
                        'cursor': 'pointer'
                    }
                )
            ], style={'marginBottom': '15px'}))
            
            # Add label showing which words are selected
            if selected_words and len(selected_words) > 0:
                selected_span = [
                    html.Span(
                        TRANSLATIONS[language]['showing_reviews_with'] + ' ',
                        style={'fontWeight': 'bold'}
                    ),
                    html.Span(', '.join(selected_words))
                ]
            else:
                selected_span = [html.Span(
                        TRANSLATIONS[language]['showing_all_reviews'],
                        style={'fontWeight': 'bold'}
                    )
                    ]

            selected_display = html.Div(selected_span, style={
                'marginBottom': '10px', 
                'padding': '8px 10px', 
                'backgroundColor': '#e8f4ff', 
                'borderRadius': '4px',
                'border': '1px solid #b8d8ff',
                'fontSize': '14px'
            })
            word_freq_components.append(selected_display)
        
        # Filter reviews by selected words if needed (already done before calling this function?)
        # No, the filtering should happen *here* based on the selected words passed in.
        if selected_words and len(selected_words) > 0:
            word_filtered_review_dicts = []
            lower_selected_words = [w.lower() for w in selected_words]

            for review_info in filtered_review_dicts:
                if not review_info: continue

                # Check ngrams first
                all_review_ngrams_lower = [n['text'].lower() for n in review_info.get('review_ngrams', [])]
                detail_ngrams_lower = [n['text'].lower() for n in review_info.get('highlight_detail_ngrams', [])]
                reason_ngrams_lower = [n['text'].lower() for n in review_info.get('highlight_reason_ngrams', [])]

                found_in_ngram = False
                for word in lower_selected_words:
                    if word in all_review_ngrams_lower or word in detail_ngrams_lower or word in reason_ngrams_lower:
                        found_in_ngram = True
                        break

                if found_in_ngram:
                    word_filtered_review_dicts.append(review_info)
                    continue # Go to next review if found in ngram

                # If not found in ngrams, check text fields (less precise)
                # This part might be redundant if frequent words are primarily ngrams
                # Let's keep it as a fallback for now.
                full_text_lower = review_info.get('full_review_text', '').lower()
                detail_text_lower = review_info.get('highlight_detail_text', '').lower()
                reason_text_lower = (review_info.get('highlight_reason_text', '') or '').lower()

                found_in_text = False
                for word in lower_selected_words:
                    if word in full_text_lower or word in detail_text_lower or word in reason_text_lower:
                        found_in_text = True
                        break

                if found_in_text:
                    word_filtered_review_dicts.append(review_info)

            final_review_dicts_to_display = word_filtered_review_dicts

            # If no reviews match after word filtering, add a message
            if not final_review_dicts_to_display:
                return (
                    word_freq_components +
                    [html.P(TRANSLATIONS[language]['no_reviews_matching_words'])]
                )
        else:
            # If no words selected, display all originally passed reviews
            final_review_dicts_to_display = filtered_review_dicts

        # Create HTML for reviews
        reviews_html = []
        for idx, review_info in enumerate(final_review_dicts_to_display):
            review_html_line = format_review_html(review_info, selected_words) # Pass selected words for highlighting
            reviews_html.append(f"""
                <div class="review-container">
                    <span style='color: #555; font-size: 0.9em;'>{TRANSLATIONS[language]['review']} {idx + 1}: </span>
                    {review_html_line}
                </div>
            """)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    margin: 0;
                    font-family: sans-serif;
                    line-height: 1.4;
                }}
                .review-container {{
                    margin: 0;
                    padding: 3px;
                    border-bottom: 1px solid #eee;
                    word-wrap: break-word; /* Ensure long words/IDs wrap */
                    overflow-wrap: break-word;
                }}
                .review-container:last-child {{
                    border-bottom: none;
                }}
                b {{ /* Style for title tag */
                    font-weight: bold;
                 }}
                 /* Style for selected ngram highlights */
                 span[style*="background-color: #FFFF00"] {{
                    padding: 1px 0; /* Add slight padding for visual separation */
                }}
            </style>
        </head>
        <body>
            {''.join(reviews_html)}
        </body>
        </html>
        """
        
        # Combine word frequency components with reviews iframe
        result = []
        # Add header HTML after word frequency components
        # Format categories before passing to header
        formatted_x = format_category_display(x_category, language) if x_category else None
        formatted_y = format_category_display(y_category, language) if y_category else None
        header_html = create_header_html(language, plot_type, formatted_x, formatted_y)
        result.append(html.Iframe(
            srcDoc=header_html,
            style={
                'width': '100%',
                'height': '90px',
                'border': 'none',
                'borderRadius': '3px',
                'backgroundColor': 'white',
                'marginBottom': '10px'
            }
        ))
        if word_freq_components:
            # Wrap word frequency components in a div for better layout control
            result.append(html.Div(word_freq_components, style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'marginBottom': '10px'}))
        
        # Add the reviews in an iframe
        result.append(html.Iframe(
            srcDoc=html_content,
            style={
                'width': '100%',
                'height': '630px', # Adjust height as needed
                'border': '1px solid #ddd', # Add subtle border
                'borderRadius': '3px',
                'backgroundColor': 'white'
            }
        ))
        
        return result

    def create_header_html(language, plot_type, x_category=None, y_category=None):
        """Create header HTML with selection info"""
        # Use the appropriate lambda function based on plot type
        if plot_type == 'bar_chart':
            review_format = TRANSLATIONS[language]['get_bar_chart_review_format'](x_category)
        else:  # matrix
            review_format = TRANSLATIONS[language]['get_matrix_review_format'](x_category, y_category)
        
        return f"""
        <html>
        <head>
            <style>
                body {{
                    margin: 5px;
                    font-family: sans-serif;
                    line-height: 1.4;
                    padding: 0px;
                    border-bottom: 1px ridge #eee;
                }}
                span {{
                    display: inline-block;
                    margin: 2px;
                }}
            </style>
        </head>
        <body>
            <div style="margin-top: 3px;">
                {review_format}
            </div>
        </body>
        </html>
        """

    def create_count_display(positive_reviews, negative_reviews, sentiment_filter, language):
        """Create count display based on filter selection"""
        if sentiment_filter == 'show_all':
            return [
                html.Span(f"{TRANSLATIONS[language]['positive_count'].format(len(positive_reviews))} | "),
                html.Span(TRANSLATIONS[language]['negative_count'].format(len(negative_reviews)))
            ]
        elif sentiment_filter == 'show_positive':
            return [html.Span(TRANSLATIONS[language]['positive_count'].format(len(positive_reviews)))]
        else:  # show_negative
            return [html.Span(TRANSLATIONS[language]['negative_count'].format(len(negative_reviews)))]
            
    def get_filter_style():
        """Return consistent filter style"""
        return {
            'display': 'block',
            'marginBottom': '5px',
            'whiteSpace': 'nowrap',
            'display': 'flex',
            'flexDirection': 'row',
            'gap': '15px',
            'alignItems': 'center'
        }

    def update_reviews_with_filters(click_data, sentiment_filter, language, plot_type,
                                  x_value, y_value, top_n_x, top_n_y, search_query,
                                  bar_categories, bar_zoom, bar_count, start_date, end_date, selected_words=None):
        """Helper function to update reviews display with sentiment and word filtering"""
        if not click_data:
            return dash.no_update, None, None, None
            
        search_query = search_query if search_query else ''
        selected_words = selected_words if selected_words else []
        
        # Handle date filter
        if not isinstance(start_date, str) and not isinstance(end_date, str):
            start_date = None
            end_date = None
        
        # Logic for bar chart
        if plot_type == 'bar_chart':
            # Get data for bar chart
            if not bar_categories:
                return dash.no_update, None, None, None
                
            if bar_zoom == '':
                bar_zoom = None

            # review_data is now a list of lists of review dictionaries
            display_categories, original_categories, counts, sentiment_ratios, review_data_outer, colors, title_key = get_bar_chart_data(
                bar_categories, bar_zoom, language, search_query, start_date, end_date
            )
            
            if bar_count > 0 and bar_count < len(display_categories):
                display_categories = display_categories[:bar_count]
                original_categories = original_categories[:bar_count]
                counts = counts[:bar_count]
                sentiment_ratios = sentiment_ratios[:bar_count]
                review_data_outer = review_data_outer[:bar_count]
                colors = colors[:bar_count]
            
            # Process click data - accepting any valid curveNumber
            if 'points' in click_data and len(click_data['points']) > 0 and 'label' in click_data['points'][0]:
                clicked_label = click_data['points'][0]['label']
                
                # Create mapping between displayed category and internal index
                formatted_to_original = {}
                
                # Use proper category formatting to match displayed label
                prefix = None
                is_zoomed = bar_zoom and bar_zoom != 'all'
                
                # Check for category prefix
                for cat_prefix in ['U: ', 'A: ', 'P: ']:
                    if clicked_label.startswith(cat_prefix):
                        prefix = cat_prefix
                        break
                
                # Map all displayed categories to their indices
                for i, category in enumerate(display_categories):
                    # Special case for zoomed categories
                    if is_zoomed and '|' in category:
                        parts = category.split('|')
                        display_text = '|'.join(parts[1:])
                    else:
                        display_text = category
                        display_text = format_category_display(display_text, language)

                    formatted_to_original[display_text] = i

                    # Also map intermediate parts if hierarchical and zoomed
                    if is_zoomed and '|' in category:
                        parts = category.split('|')
                        for k in range(1, len(parts)):
                            sub_cat_display = format_category_display('|'.join(parts[k:]), language)
                            if sub_cat_display not in formatted_to_original:
                                formatted_to_original[sub_cat_display] = i
                    else:
                        # Map the full formatted category name as well
                        full_formatted_category_name = format_category_display(category, language)
                        if full_formatted_category_name not in formatted_to_original:
                             formatted_to_original[full_formatted_category_name] = i
                
                # Try finding the index based on raw category name first
                idx = None
                
                # Direct match first
                if clicked_label in display_categories:
                    idx = display_categories.index(clicked_label)
                else:
                    # Try finding by formatted category name
                    for formatted_cat, index in formatted_to_original.items():
                        # Try to match with and without the prefix
                        if formatted_cat == clicked_label or (prefix and formatted_cat == clicked_label[len(prefix):]):
                            idx = index
                            break
                
                    # If still not found, try raw string match with display categories
                    if idx is None:
                        for i, cat in enumerate(display_categories):
                            if cat == clicked_label or (prefix and cat[len(prefix):] == clicked_label[len(prefix):]):
                                idx = i
                                break

                if idx is None:
                    # Try direct index from pointNumber as last resort
                    try:
                        point_number = click_data['points'][0]['pointNumber']
                        if 0 <= point_number < len(display_categories):
                            idx = point_number
                        else:
                            return dash.no_update, None, None, None
                    except (KeyError, IndexError):
                        return dash.no_update, None, None, None

                # Get the list of review dictionaries for the clicked bar
                review_dicts = review_data_outer[idx]
                
                # Check if these are already the right format or if we need to extract 'review_info'
                if review_dicts and isinstance(review_dicts[0], dict) and 'sentiment' in review_dicts[0] and 'review_info' in review_dicts[0]:
                    # Extract only the review_info part
                    review_infos = [item['review_info'] for item in review_dicts if 'review_info' in item]
                else:
                    # Already in the right format
                    review_infos = review_dicts

                # Apply sentiment filtering
                positive_reviews, negative_reviews, neutral_reviews = categorize_reviews(review_infos)
                filtered_reviews = filter_reviews_by_sentiment(review_infos, positive_reviews, negative_reviews, sentiment_filter)

                # Create updated review display with word filtering
                reviews_display = create_review_display(
                    filtered_reviews,
                    language,
                    plot_type='bar_chart',
                    x_category=display_categories[idx], # Pass the category name for the header
                    selected_words=selected_words
                )
                
                # Return updated reviews display and sentiment counts 
                return reviews_display, positive_reviews, negative_reviews, sentiment_filter
            else:
                return dash.no_update, None, None, None
        else:
            # Handle matrix view selection
            point = click_data['points'][0]
            clicked_x = point['x']
            clicked_y = point['y']
            
            matrix, sentiment_matrix, review_matrix, x_text, x_percentages, y_text, y_percentages, title_key = get_plot_data(
                plot_type, x_value, y_value, top_n_x, top_n_y, language, search_query, start_date, end_date
            )
            
            # Format display labels
            formatted_x_display = [format_category_display(label, language) for label in x_text]
            formatted_y_display = [format_category_display(label, language) for label in y_text]
            
            # Create mappings
            x_mapping = {formatted: i for i, formatted in enumerate(formatted_x_display)}
            y_mapping = {formatted: i for i, formatted in enumerate(formatted_y_display)}
            
            try:
                # Find the indices
                i = y_mapping.get(clicked_y)
                j = x_mapping.get(clicked_x)
                
                # Try lenient matching if not found
                if i is None:
                    for formatted_y, idx in y_mapping.items():
                        if clicked_y in formatted_y or formatted_y in clicked_y:
                            i = idx
                            break
                
                if j is None:
                    for formatted_x, idx in x_mapping.items():
                        if clicked_x in formatted_x or formatted_x in clicked_x:
                            j = idx
                            break
                
                if i is not None and j is not None:
                    reviews = review_matrix[i][j]
                    
                    # Apply sentiment filtering
                    positive_reviews, negative_reviews, neutral_reviews = categorize_reviews(reviews)
                    filtered_reviews = filter_reviews_by_sentiment(reviews, positive_reviews, negative_reviews, sentiment_filter)
                    
                    # Create updated review display with word filtering
                    reviews_display = create_review_display(
                        filtered_reviews, 
                        language, 
                        plot_type='matrix',
                        x_category=x_text[j],
                        y_category=y_text[i],
                        selected_words=selected_words
                    )
                    
                    return reviews_display, positive_reviews, negative_reviews, None
                    
            except ValueError as e:
                print(f"Error processing matrix click data: {str(e)}")
        
        return dash.no_update, None, None, None

    # Reviews update callback
    @app.callback(
        [Output('reviews-content', 'children', allow_duplicate=True),
         Output('sentiment-filter', 'value'),
         Output('sentiment-filter', 'style'),
         Output('review-counts', 'children'),
         Output('selected-words-storage', 'children', allow_duplicate=True)],
        [Input('sentiment-filter', 'value'),
         Input('main-figure', 'clickData'),
         Input('app-language-state', 'children')],
        [State('plot-type', 'value'),
         State('x-axis-dropdown', 'value'),
         State('y-axis-dropdown', 'value'),
         State('x-features-slider', 'value'),
         State('y-features-slider', 'value'),
         State('search-input', 'value'),
         State('bar-category-checklist', 'value'),
         State('bar-zoom-dropdown', 'value'),
         State('bar-count-slider', 'value'),
         State('date-filter-storage', 'children'),
         State('selected-words-storage', 'children')],
        prevent_initial_call=True
    )
    def update_reviews_with_sentiment_filter(sentiment_filter, click_data, language, 
                                        plot_type, x_value, y_value, top_n_x, top_n_y, 
                                        search_query, bar_categories, bar_zoom, bar_count, 
                                        date_filter_storage, selected_words_json):
    
        # Parse date range from storage
        date_range = json.loads(date_filter_storage) if date_filter_storage else {"start_date": None, "end_date": None}
        start_date = date_range.get("start_date")
        end_date = date_range.get("end_date")
        
        # Default responses
        hide_style = {'display': 'none'}
        show_style = get_filter_style()
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
        
        # Initialize review styles and variables
        reviews_content = html.Div([html.Div(TRANSLATIONS[language]['no_reviews'], style={'padding': '20px', 'textAlign': 'center'})])
        count_display = ''
        
        # Initialize review counts
        if click_data is None:
            return reviews_content, 'show_all', hide_style, count_display, '[]'
        
        # Get selected words
        selected_words = json.loads(selected_words_json) if selected_words_json else []
        # If the callback was triggered by clicking on the bar chart, clear selected words
        if dash.callback_context.triggered and str(dash.callback_context.triggered[0]['prop_id']).startswith('main-figure'):
            selected_words = []
            selected_words_json = json.dumps(selected_words)
        
        try:
            # Update reviews with current filters
            result = update_reviews_with_filters(
                click_data, sentiment_filter, language, plot_type, x_value, y_value, 
                top_n_x, top_n_y, search_query, bar_categories, bar_zoom, bar_count, 
                start_date, end_date, selected_words
            )
            
            if result is not None and len(result) == 4:
                reviews_content, pos_reviews, neg_reviews, _ = result
                
                # Create count display if we have review counts
                if pos_reviews is not None and neg_reviews is not None:
                    count_display = create_count_display(pos_reviews, neg_reviews, sentiment_filter, language)
        except Exception as e:
            print(f"Error in update_reviews_with_sentiment_filter: {str(e)}")
            # Keep default values in case of error
        
        return reviews_content, sentiment_filter, show_style, count_display, selected_words_json

    # Add callback to handle word selection
    @app.callback(
        [Output('reviews-content', 'children', allow_duplicate=True),
         Output('selected-words-storage', 'children', allow_duplicate=True)],
        [Input({'type': 'word-button', 'index': dash.dependencies.ALL}, 'n_clicks'),
         Input('clear-word-selection-button', 'n_clicks')],
        [State('selected-words-storage', 'children'),
         State('sentiment-filter', 'value'),
         State('main-figure', 'clickData'),
         State('app-language-state', 'children'),
         State('plot-type', 'value'),
         State('x-axis-dropdown', 'value'),
         State('y-axis-dropdown', 'value'),
         State('x-features-slider', 'value'),
         State('y-features-slider', 'value'),
         State('search-input', 'value'),
         State('bar-category-checklist', 'value'),
         State('bar-zoom-dropdown', 'value'),
         State('bar-count-slider', 'value'),
         State('date-filter-storage', 'children')],
        prevent_initial_call=True
    )
    def update_reviews_by_word_selection(word_buttons_clicks, clear_clicks, 
                                        selected_words_json, sentiment_filter, click_data, 
                                        language, plot_type, x_value, y_value, top_n_x, top_n_y, 
                                        search_query, bar_categories, bar_zoom, bar_count, date_filter_storage):
        """Update the displayed reviews based on word selection"""
        # Check which input triggered the callback
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
        
        # Convert selected_words_json to list
        selected_words = json.loads(selected_words_json) if selected_words_json else []
        
        # Get all available words from the word buttons
        all_words = []
        for item in ctx.inputs_list[0]:
            if 'id' in item and isinstance(item['id'], dict) and 'index' in item['id']:
                all_words.append(item['id']['index'])
        
        # Handle different triggers
        if 'clear-word-selection-button' in trigger_id:
            selected_words = []
        elif 'word-button' in trigger_id and sum(word_buttons_clicks) > 0:
            try:
                # Try to parse the trigger ID as JSON
                trigger_dict = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
                if trigger_dict.get('type') == 'word-button':
                    clicked_word = trigger_dict.get('index')
                    # Toggle the word selection
                    if clicked_word in selected_words:
                        selected_words.remove(clicked_word)
                    else:
                        selected_words.append(clicked_word)
            except json.JSONDecodeError:
                pass
        
        # Parse date range from storage
        date_range = json.loads(date_filter_storage) if date_filter_storage else {"start_date": None, "end_date": None}
        start_date = date_range.get("start_date")
        end_date = date_range.get("end_date")
        
        # Update reviews with the current word selection
        reviews_content, _, _, _ = update_reviews_with_filters(
            click_data, sentiment_filter, language, plot_type, x_value, y_value, 
            top_n_x, top_n_y, search_query, bar_categories, bar_zoom, bar_count, 
            start_date, end_date, selected_words
        )
        
        return reviews_content, json.dumps(selected_words)

    # Search-related callbacks
    @app.callback(
        Output('search-help-tooltip', 'children'),
        [Input('app-language-state', 'children')]
    )
    def update_search_help_tooltip(language):
        return get_search_examples_html(language)
    
    @app.callback(
        Output('search-help-tooltip', 'style'),
        [Input('search-help-button', 'n_clicks')],
        [State('search-help-tooltip', 'style')]
    )
    def toggle_search_help(n_clicks, current_style):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
            
        # Create a copy of the current style
        new_style = dict(current_style)
        
        # Toggle the display property
        if new_style.get('display') == 'none':
            new_style['display'] = 'block'
        else:
            new_style['display'] = 'none'
            
        return new_style
    
    # Add a clientside callback to hide tooltip when clicking elsewhere
    app.clientside_callback(
        """
        function(n_clicks) {
            var tooltipElement = document.getElementById('search-help-tooltip');
            
            document.addEventListener('click', function(event) {
                var isClickInside = document.getElementById('search-help-button').contains(event.target);
                
                if (!isClickInside && tooltipElement.style.display !== 'none') {
                    tooltipElement.style.display = 'none';
                }
            });
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('search-help-button', 'n_clicks', allow_duplicate=True),
        [Input('search-help-button', 'id')],
        prevent_initial_call=True
    )

    @app.callback(
        Output('search-input', 'placeholder'),
        [Input('app-language-state', 'children')]
    )
    def update_search_placeholder(language):
        return TRANSLATIONS[language]['search_placeholder']

    @app.callback(
        Output('search-button', 'n_clicks'),
        [Input('search-input', 'n_submit')],
        [State('search-button', 'n_clicks')]
    )
    def handle_enter_press(n_submit, current_clicks):
        if n_submit:
            return (current_clicks or 0) + 1
        return dash.no_update

    @app.callback(
        Output('search-results-info', 'children'),
        [Input('search-button', 'n_clicks'),
        Input('app-language-state', 'children'),
        Input('date-filter-slider', 'value')],
        [State('search-input', 'value'),
        State('plot-type', 'value'),
        State('x-axis-dropdown', 'value'),
        State('y-axis-dropdown', 'value'),
        State('date-filter-storage', 'children')]
    )
    def update_search_results_info(n_clicks, language, date_slider_value, search_query, plot_type, x_value, y_value, date_filter_storage):
        """Display information about search results."""
        if n_clicks == 0 or not search_query or not search_query.strip():
            return ""
            
        # Parse date range from storage
        date_range = json.loads(date_filter_storage) if date_filter_storage else {"start_date": None, "end_date": None}
        start_date = date_range.get("start_date")
        end_date = date_range.get("end_date")
        
        # Count reviews that match the search query
        try:
            # Handle different plot types
            if plot_type == 'bar_chart':
                # For bar chart, count reviews across all categories
                categories = ['usage', 'attribute', 'performance']
                total_reviews = 0
                for category_type in categories:
                    dict_type = {
                        'usage': 'use_sents',
                        'attribute': 'attr_sents',
                        'performance': 'perf_sents'
                    }[category_type]
                    
                    # Get filtered dictionary
                    filtered_dict = get_cached_dict(dict_type, 'all', search_query, start_date, end_date)
                    
                    # Count reviews
                    for path, sents_dict in filtered_dict.items():
                        for sent, val in sents_dict.items():
                            if isinstance(val, dict):
                                for rid, reviews in val.items():
                                    if reviews:
                                        total_reviews += len(reviews)
                            elif isinstance(val, list) and val:
                                total_reviews += len(val)
            
            else:
                # For matrix views, count reviews matching the plot type
                if plot_type == 'use_attr_perf':
                    x_dict_type = 'use_sents'
                    y_dict_type = 'attr_perf_sents'
                else:  # perf_attr
                    x_dict_type = 'perf_sents'
                    y_dict_type = 'attr_sents'
                    
                # Get filtered dictionaries
                x_dict = get_cached_dict(x_dict_type, x_value or 'all', search_query, start_date, end_date)
                y_dict = get_cached_dict(y_dict_type, y_value or 'all', search_query, start_date, end_date)
                
                # Count reviews
                total_reviews = 0
                for path, sents_dict in x_dict.items():
                    for sent, val in sents_dict.items():
                        if isinstance(val, dict):
                            for rid, reviews in val.items():
                                if reviews:
                                    total_reviews += len(reviews)
                        elif isinstance(val, list) and val:
                            total_reviews += len(val)
                            
                for path, sents_dict in y_dict.items():
                    for sent, val in sents_dict.items():
                        if isinstance(val, dict):
                            for rid, reviews in val.items():
                                if reviews:
                                    total_reviews += len(reviews)
                        elif isinstance(val, list) and val:
                            total_reviews += len(val)
            
            # Format message
            if total_reviews > 0:
                normalized_query = normalize_search_query(search_query)
                return TRANSLATIONS[language]['search_results'].format(total_reviews, normalized_query)
            else:
                return "No reviews found matching your search criteria."
        except Exception as e:
            print(f"Error counting search results: {str(e)}")
            return "Error processing search query."

    @app.callback(
        Output('sentiment-filter', 'options'),
        [Input('app-language-state', 'children')]
    )
    def update_sentiment_filter_options(language):
        return [
            {'label': TRANSLATIONS[language]['show_all'], 'value': 'show_all'},
            {'label': TRANSLATIONS[language]['show_positive'], 'value': 'show_positive'},
            {'label': TRANSLATIONS[language]['show_negative'], 'value': 'show_negative'}
        ]

    # Add a callback to display current selected date range
    @app.callback(
        Output('date-range-display', 'children'),
        [Input('date-filter-slider', 'value'),
         Input('date-filter-storage', 'children'),
         Input('app-language-state', 'children'),
         Input('total-reviews-count', 'data')]
    )
    def update_date_display(slider_value, date_filter_storage, language, total_reviews):
        if not slider_value or not date_filter_storage:
            return ""
            
        try:
            # Parse the date storage
            storage = json.loads(date_filter_storage)
            min_date = storage.get("min_date")
            
            if not min_date:
                return ""
                
            # Convert slider values to dates
            min_date_obj = datetime.strptime(min_date, '%Y-%m-%d')
            start_date_obj = min_date_obj + timedelta(days=slider_value[0])
            end_date_obj = min_date_obj + timedelta(days=slider_value[1])
            
            # Format as YYYY-MM for display
            start_date_display = start_date_obj.strftime('%Y-%m')
            end_date_display = end_date_obj.strftime('%Y-%m')
            
            # For actual filtering, we still need the full date
            start_date = start_date_obj.strftime('%Y-%m-%d')
            end_date = end_date_obj.strftime('%Y-%m-%d')
            
            # Calculate the span in months for additional info
            month_diff = (end_date_obj.year - start_date_obj.year) * 12 + (end_date_obj.month - start_date_obj.month)
            
            if month_diff > 0:
                month_label = TRANSLATIONS[language].get('months', 'months')
                
            return html.Div([
                
                html.Div([
                    html.Button(
                        TRANSLATIONS[language].get('selected_date_range'),
                        id='apply-date-btn',
                        style={'fontSize': '14px', 'padding': '2px 8px', 'marginRight': '10px'}
                    ),
                    dcc.Input(
                        id='start-date-input',
                        type='text',
                        value=start_date_display,
                        style={'width': '80px', 'textAlign': 'center', 'color': '#2196F3', 'fontWeight': 'bold'}
                    ),
                    html.Span(" → ", style={'margin': '0 10px'}),
                    dcc.Input(
                        id='end-date-input',
                        type='text',
                        value=end_date_display,
                        style={'width': '80px', 'textAlign': 'center', 'color': '#2196F3', 'fontWeight': 'bold'}
                    ),
                    html.Span(f" ({month_diff+1} {month_label}, {TRANSLATIONS[language].get('total_reviews', 'Total reviews')}: {total_reviews})"),
                    
                ], style={'marginTop': '5px'})
            ])
        except Exception as e:
            print(f"Error updating date display: {str(e)}")
            return ""
            
    @app.callback(
        [Output('date-filter-slider', 'value', allow_duplicate=True),
         Output('date-filter-storage', 'children', allow_duplicate=True)],
        [Input('apply-date-btn', 'n_clicks')],
        [State('start-date-input', 'value'),
         State('end-date-input', 'value'),
         State('date-filter-storage', 'children')],
        prevent_initial_call=True
    )
    def update_date_from_input(n_clicks, start_date_input, end_date_input, date_filter_storage):
        if not n_clicks or not date_filter_storage:
            raise dash.exceptions.PreventUpdate
            
        try:
            # Parse current date storage
            storage_data = json.loads(date_filter_storage)
            min_date_str = storage_data.get("min_date")
            max_date_str = storage_data.get("max_date")
            
            if not min_date_str or not max_date_str:
                raise dash.exceptions.PreventUpdate
                
            # Parse base min date
            min_date = datetime.strptime(min_date_str, '%Y-%m-%d')
            
            # Parse user input dates
            # Add day component if missing
            if len(start_date_input) <= 7:  # Format is YYYY-MM
                start_date_input = f"{start_date_input}-01"
            if len(end_date_input) <= 7:  # Format is YYYY-MM
                # Set to last day of the month
                year, month = end_date_input.split('-')
                last_day = pd.Period(f"{year}-{month}").days_in_month
                end_date_input = f"{end_date_input}-{last_day}"
                
            try:
                # Try to parse with different formats
                try:
                    start_date = datetime.strptime(start_date_input, '%Y-%m-%d')
                except ValueError:
                    start_date = datetime.strptime(start_date_input, '%Y-%m')
                    start_date = datetime(start_date.year, start_date.month, 1)
                    
                try:
                    end_date = datetime.strptime(end_date_input, '%Y-%m-%d')
                except ValueError:
                    end_date = datetime.strptime(end_date_input, '%Y-%m')
                    # Set to last day of month
                    if end_date.month == 12:
                        end_date = datetime(end_date.year + 1, 1, 1) - timedelta(days=1)
                    else:
                        end_date = datetime(end_date.year, end_date.month + 1, 1) - timedelta(days=1)
            except ValueError:
                # If parsing fails, show an alert and don't update
                print(f"Invalid date format: {start_date_input} or {end_date_input}")
                raise dash.exceptions.PreventUpdate
                
            # Calculate slider values
            start_slider_value = (start_date - min_date).days
            end_slider_value = (end_date - min_date).days
            
            # Ensure dates are within valid range
            max_date = datetime.strptime(max_date_str, '%Y-%m-%d')
            start_slider_value = max(0, min(start_slider_value, (max_date - min_date).days))
            end_slider_value = max(start_slider_value, min(end_slider_value, (max_date - min_date).days))
            
            # Update storage
            storage_data["start_date"] = start_date.strftime('%Y-%m-%d')
            storage_data["end_date"] = end_date.strftime('%Y-%m-%d')
            
            return [start_slider_value, end_slider_value], json.dumps(storage_data)
            
        except Exception as e:
            print(f"Error updating dates from input: {str(e)}")
            raise dash.exceptions.PreventUpdate

    # Add clientside callback to handle pathname changes
    app.clientside_callback(
        """
        function(pathname) {
            if (pathname === "/login") {
                return [{'display': 'none'}, {'display': 'block'}];
            } else if (pathname === "/") {
                return [{'display': 'block'}, {'display': 'none'}];
            } else {
                return [{'display': 'none'}, {'display': 'block'}];
            }
        }
        """,
        [Output('main-content', 'style'),
         Output('login-content', 'style')],
        [Input('url', 'pathname')],
    )

    @app.callback(
        Output('trend-chart-container', 'style'),
        [Input('plot-type', 'value')]
    )
    def toggle_trend_chart_visibility(plot_type):
        """Show trend chart only for bar chart visualization"""
        if plot_type == 'bar_chart':
            return {'display': 'block'}
        else:
            return {'display': 'none'}

    # Update the trend chart callback
    @app.callback(
        Output('trend-chart', 'figure'),
        [Input('plot-type', 'value'),
         Input('bar-category-checklist', 'value'),
         Input('bar-zoom-dropdown', 'value'),
         Input('bar-count-slider', 'value'),
         Input('app-language-state', 'children'),
         Input('search-button', 'n_clicks'),
         Input('date-filter-slider', 'value'),
         Input('time-bucket-dropdown', 'value')],
        [State('search-input', 'value'),
         State('date-filter-storage', 'children')]
    )
    def update_trend_chart(plot_type, bar_categories, bar_zoom_value, bar_count, language, n_clicks, 
                          date_slider_value, time_bucket, search_query, date_filter_storage):
        """
        Update the trend chart showing category mentions over time.
        """
        # Check which input triggered the callback
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    
        
        # Only create trend chart for bar chart view
        if plot_type != 'bar_chart':
            # Return an empty figure
            return go.Figure()

        # Handle empty values with defaults
        if not bar_categories:
            bar_categories = ['usage', 'attribute', 'performance']
        if bar_zoom_value is None or bar_zoom_value == '':
            bar_zoom_value = 'all'
        if time_bucket is None:
            time_bucket = '3month'
        
        # Parse date range from storage
        date_range = json.loads(date_filter_storage) if date_filter_storage else {"start_date": None, "end_date": None}
        start_date = date_range.get("start_date")
        end_date = date_range.get("end_date")
        
        # Get bar chart data for categories
        display_categories, original_categories, counts, sentiment_ratios, review_data, colors, title_key = get_bar_chart_data(
            bar_categories, bar_zoom_value, language, search_query, start_date, end_date
        )
        
        # Apply bar count limit from bar_count slider for visualization
        if bar_count > 0 and bar_count < len(display_categories):
            display_categories = display_categories[:bar_count]
            original_categories = original_categories[:bar_count]
            
        # Get time series data for the trend chart
        # Note: We pass the original display_categories with prefixes
        time_series_data = get_category_time_series(
            bar_categories, display_categories, original_categories, 
            bar_zoom_value, language, search_query, start_date, end_date, 
            time_bucket=time_bucket
        )
        
        # Create the trend chart - initially show all lines
        selected_trend_lines = [cat for cat in display_categories 
                              if cat in time_series_data['category_data'] and 
                                 any(count > 0 for count in time_series_data['category_data'][cat]['counts'])]
        
        # If we have no trend lines with data, check if there's an issue with missing prefixes
        if not selected_trend_lines and display_categories:
            # Create a fallback figure with an informative message
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text="No time series data available for the selected categories",
                font=dict(size=16),
                showarrow=False,
                xref="paper", yref="paper"
            )
            return fig
            
        fig = create_trend_chart(display_categories, original_categories, time_series_data, language, selected_trend_lines)
        
        return fig

    @app.callback(
        Output('time-bucket-label', 'children'),
        [Input('app-language-state', 'children')]
    )
    def update_time_bucket_label(language):
        return TRANSLATIONS[language].get('time_bucket_label', 'Time interval:')

    @app.callback(
        Output('trend-line-selector-label', 'children'),
        [Input('app-language-state', 'children')]
    )
    def update_trend_line_selector_label(language):
        return TRANSLATIONS[language].get('trend_line_selector_label', 'Select trend lines to display:')

    # Callback to update trend line selector options based on available categories
    @app.callback(
        [Output('trend-line-selector', 'options'),
         Output('trend-line-selector', 'value')],
        [Input('plot-type', 'value'),
         Input('bar-category-checklist', 'value'),
         Input('bar-zoom-dropdown', 'value'),
         Input('bar-count-slider', 'value'),
         Input('app-language-state', 'children'),
         Input('date-filter-slider', 'value')],
        [State('search-input', 'value'),
         State('date-filter-storage', 'children'),
         State('time-bucket-dropdown', 'value')]
    )
    def update_trend_line_selector(plot_type, bar_categories, bar_zoom_value, bar_count, language, 
                                  date_slider_value, search_query, date_filter_storage, time_bucket):
        # Only update for bar chart view
        if plot_type != 'bar_chart':
            return [], []

        # Handle empty values with defaults
        if not bar_categories:
            bar_categories = ['usage', 'attribute', 'performance']
        if bar_zoom_value is None:
            bar_zoom_value = 'all'
        if time_bucket is None:
            time_bucket = '3month'
        
        # Parse date range from storage
        date_range = json.loads(date_filter_storage) if date_filter_storage else {"start_date": None, "end_date": None}
        start_date = date_range.get("start_date")
        end_date = date_range.get("end_date")
        
        # Get bar chart data for categories
        display_categories, original_categories, counts, sentiment_ratios, review_data, colors, title_key = get_bar_chart_data(
            bar_categories, bar_zoom_value, language, search_query, start_date, end_date
        )
        
        # Apply bar count limit from bar_count slider
        if bar_count > 0 and bar_count < len(display_categories):
            display_categories = display_categories[:bar_count]
            original_categories = original_categories[:bar_count]
        
        # Get time series data to check which categories actually have data
        time_series_data = get_category_time_series(
            bar_categories, display_categories, original_categories, bar_zoom_value,
            language, search_query, start_date, end_date, time_bucket=time_bucket
        )
        
        # Create options and set all as initially selected
        valid_options = []
        initial_values = []
        
        for category in display_categories:
            # Check if category has data in the time series
            if (category in time_series_data['category_data'] and 
                any(count > 0 for count in time_series_data['category_data'][category]['counts'])):
                # Determine color for colored label (similar to trend chart)
                prefix = None
                for p in type_colors.keys():
                    if category.startswith(p):
                        prefix = p
                        break
                
                # Format the label similar to other parts of the app
                formatted_category = format_category_display(category, language)
                
                valid_options.append({
                    'label': formatted_category,
                    'value': category
                })
                initial_values.append(category)
        
        return valid_options, initial_values

    @app.callback(
        Output('time-bucket-dropdown', 'options'),
        [Input('app-language-state', 'children')]
    )
    def update_time_bucket_options(language):
        return [
            {'label': TRANSLATIONS[language].get('month', 'Month'), 'value': 'month'},
            {'label': TRANSLATIONS[language].get('three_month', '3 Months'), 'value': '3month'},
            {'label': TRANSLATIONS[language].get('six_month', '6 Months'), 'value': '6month'},
            {'label': TRANSLATIONS[language].get('year', 'Year'), 'value': 'year'}
        ]

    # Add URL pathname callback for login vs main content display
    @app.callback(
        [Output('login-content', 'style', allow_duplicate=True),
         Output('main-content', 'style', allow_duplicate=True)],
        [Input('url', 'pathname')],
        [State('app-language-state', 'children'),
         State('category-state', 'children')],
        prevent_initial_call=True
    )
    def display_page(pathname, language, category):
        """Display login page or main content based on URL pathname and authentication status."""
        # Check if the user is authenticated
        from flask_login import current_user
        
        # Main logic: show login page unless authenticated and on main page
        if pathname == '/' and current_user.is_authenticated:
            return {'display': 'none'}, {'display': 'block'}  # Show main content
        else:
            return {'display': 'block'}, {'display': 'none'}  # Show login page

    # Add callback to update login page content based on language
    @app.callback(
        Output('login-content', 'children'),
        [Input('app-language-state', 'children')]
    )
    def update_login_content(language):
        """Update login page content based on the selected language."""
        from layouts import get_login_layout
        return get_login_layout(language)

    # Add callback to update main page content based on language
    @app.callback(
        Output('main-content', 'children'),
        [Input('app-language-state', 'children')]
    )
    def update_main_content(language):
        """Update main page content based on the selected language."""
        from layouts import get_main_layout
        return get_main_layout(language)

    @app.callback(
        [Output('url', 'pathname', allow_duplicate=True),
         Output('url', 'search', allow_duplicate=True)],
        [Input('url', 'pathname')],
        [State('url', 'search'),
         State('app-language-state', 'children'),
         State('category-state', 'children')],
        prevent_initial_call=True
    )
    def check_auth_and_redirect(pathname, search, language, category):
        """Check user authentication and redirect if necessary."""
        from flask_login import current_user
        
        # If user is not authenticated and tries to access any page other than login, redirect to login
        if pathname != '/login' and not current_user.is_authenticated:
            return '/login', f'?lang={language}&category={category}'
            
        # If user is authenticated and tries to access login page, redirect to main page
        if pathname == '/login' and current_user.is_authenticated:
            return '/', f'?lang={language}&category={category}'
            
        # Otherwise, don't change the URL
        raise dash.exceptions.PreventUpdate

def create_trend_chart(display_categories, original_categories, time_series_data, language, selected_trend_lines=None):
    """Create a trend line chart figure based on the time series data."""
    fig = go.Figure()
    
    # Check if we have any dates
    if not time_series_data['dates']:
        # Create an empty figure with a message
        fig.add_annotation(
            x=0.5, y=0.5,
            text=TRANSLATIONS[language].get('no_data_available', 'No time series data available'),
            font=dict(size=16),
            showarrow=False,
            xref="paper", yref="paper"
        )
        return fig
    
    # If selected_trend_lines is None, select all lines by default
    if selected_trend_lines is None:
        selected_trend_lines = [cat for cat in display_categories 
                              if cat in time_series_data['category_data'] and 
                                 any(count > 0 for count in time_series_data['category_data'][cat]['counts'])]
    
    # Generate end dates for each time bucket
    start_dates = time_series_data['dates']
    end_dates = []
    
    # Calculate end dates based on the time bucket
    for start_date_str in start_dates:
        start_date = datetime.strptime(start_date_str, '%Y-%m')
        
        # Get the time bucket from the data
        time_bucket = time_series_data.get('time_bucket', 'month')
        
        # Calculate end date based on time bucket
        if time_bucket == 'year':
            end_date = start_date + relativedelta(years=1, days=-1)
        elif time_bucket == '6month':
            end_date = start_date + relativedelta(months=6, days=-1)
        elif time_bucket == '3month':
            end_date = start_date + relativedelta(months=3, days=-1)
        else:  # month
            if start_date.month == 12:
                end_date = datetime(start_date.year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(start_date.year, start_date.month + 1, 1) - timedelta(days=1)
                
        end_dates.append(end_date.strftime('%Y-%m'))
    
    # Track the maximum y-value for setting the y-axis range
    max_y_value = 0
    
    # Track categories with data for legend buttons
    valid_categories = []
    
    # Create the trend lines with markers for each selected category
    for i, category in enumerate(display_categories):
        # Skip if no data for this category
        if category not in time_series_data['category_data']:
            continue
            
        category_data = time_series_data['category_data'][category]
        dates = time_series_data['dates']
        counts = category_data['counts']
        sentiments = category_data['sentiment']
        
        # Skip if all counts are zero
        if all(count == 0 for count in counts):
            continue
            
        # Add to valid categories list
        valid_categories.append(category)
        
        # Track maximum count for y-axis scaling (only for visible series)   
        if category in selected_trend_lines:
            category_max = max(counts)
            if category_max > max_y_value:
                max_y_value = category_max
            
        # Determine color for this category (use same logic as bar chart)
        prefix = None
        for p in type_colors.keys():
            if category.startswith(p):
                prefix = p
                break
        
        line_color = type_colors.get(prefix, '#888888')  # Default gray if prefix not found
        
        # Create marker colors list for sentiment visualization
        marker_colors = []
        for idx, (count, sentiment) in enumerate(zip(counts, sentiments)):
            if count > 0:
                # Use the sentiment color palette - same as bar chart
                marker_colors.append(ratio_to_rgb(sentiment))
            else:
                # For zero-count points, use transparent
                marker_colors.append('rgba(0,0,0,0)')  # Transparent
        
        # Prepare date range texts for hover information
        date_ranges = [f"{start} to {end}" for start, end in zip(dates, end_dates)]
        
        # Create custom data array with both date ranges and sentiments
        custom_data = np.array(list(zip(date_ranges, sentiments)))
        
        # Format the display name for the legend with color and prefix
        formatted_category = format_category_display(category, language)
        if prefix:
            if not formatted_category.startswith(prefix):
                formatted_category = prefix + formatted_category
            legend_name = f'<span style="color:{line_color}; font-weight:bold">{formatted_category}</span>'
        else:
            legend_name = f'<span style="color:{line_color}; font-weight:bold">{formatted_category}</span>'
        
        # Add line trace with visibility based on selection
        fig.add_trace(go.Scatter(
            x=dates,
            y=counts,
            mode='lines+markers',
            name=legend_name,
            line=dict(color=line_color, width=2),
            marker=dict(
                size=10,
                color=marker_colors,
                line=dict(color=line_color, width=1)
            ),
            hovertemplate=(
                f"{TRANSLATIONS[language]['period']}: %{{customdata[0]}}<br>" +
                f"{TRANSLATIONS[language]['hover_count']}: %{{y}}<br>" +
                f"{TRANSLATIONS[language]['hover_satisfaction']}: %{{customdata[1]:.2f}}"
            ),
            customdata=custom_data,
            visible=True if category in selected_trend_lines else 'legendonly'
        ))
    
    # Add Select All / Unselect All buttons for the legend
    # Add custom buttons for selecting/unselecting all traces
    select_all_button = dict(
        args=[{'visible': [True] * len(valid_categories)}],
        label=TRANSLATIONS[language].get('select_all', "Select All"),
        method="update"
    )
    
    unselect_all_button = dict(
        args=[{'visible': ['legendonly'] * len(valid_categories)}],
        label=TRANSLATIONS[language].get('unselect_all', "Unselect All"),
        method="update"
    )
    
    # Update layout for trend chart
    fig.update_layout(
        title=TRANSLATIONS[language]['trend_chart_title'],
        xaxis_title=TRANSLATIONS[language]['trend_x_axis'],
        yaxis_title=TRANSLATIONS[language]['trend_y_axis'],
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.5)",
            borderwidth=1
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1.05,
                y=1.05,
                buttons=[select_all_button, unselect_all_button],
                pad={"r": 10, "t": 10},
                showactive=False,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.1)',
                font=dict(size=12)
            )
        ],
        height=400,
        margin=dict(
            l=80,
            r=150,
            t=100,
            b=80,
            autoexpand=True
        )
    )
    
    # Set y-axis range with some padding
    if max_y_value > 0:
        fig.update_yaxes(range=[0, max_y_value * 1.1])
    
    return fig