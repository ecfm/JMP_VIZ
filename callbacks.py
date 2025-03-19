import dash
from dash import Input, Output, State, html
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from flask_login import login_user, logout_user, current_user, UserMixin
import re

from config import TRANSLATIONS, AXIS_CATEGORY_NAMES, VALID_USERNAME, VALID_PASSWORD, get_highlight_examples, color_mapping
from data import get_plot_data, get_cached_dict, get_options, normalize_search_query, get_bar_chart_data
from utils import ratio_to_rgb, get_width_legend_translations, get_hover_translations, get_log_width, get_search_examples_html
from layouts import get_login_layout, get_main_layout

# User class for authentication
class User(UserMixin):
    def __init__(self, username):
        self.id = username

def register_callbacks(app):
    """Register all callbacks with the app."""
    
    # Authentication callbacks
    @app.callback(
        [Output('url', 'pathname'),
        Output('login-error', 'children')],
        [Input('login-button', 'n_clicks'),
        Input('username-input', 'value'),
        Input('password-input', 'value')],
        prevent_initial_call=True
    )
    def login_callback(n_clicks, username, password):
        if not n_clicks:
            return dash.no_update, dash.no_update
        
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            user = User(username)
            login_user(user)
            return '/', ''
        return dash.no_update, html.Div('Invalid credentials', style={'color': 'red'})

    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        [Input('logout-button', 'n_clicks')],
        [State('url', 'pathname')],
        prevent_initial_call=True
    )
    def logout_callback(n_clicks, pathname):
        if n_clicks is not None and n_clicks > 0:
            logout_user()
            return '/login'
        return pathname

    # Page content callback
    @app.callback(
        Output('page-content', 'children'),
        [Input('url', 'pathname'),
        Input('language-selector', 'value')]
    )
    def display_page(pathname, language='en'):
        if pathname == '/login' or not current_user.is_authenticated:
            return get_login_layout(language)
        if pathname == '/':
            return get_main_layout(language)
        return get_login_layout(language)

    # Language-dependent callbacks
    @app.callback(
        [Output('plot-type-label', 'children'),
        Output('plot-type', 'options')],
        [Input('language-selector', 'value')]
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
        [Input('language-selector', 'value')]
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
         Input('language-selector', 'value'),
         Input('search-button', 'n_clicks')],
        [State('search-input', 'value')]
    )
    def update_bar_slider_max(bar_categories, bar_zoom, language, n_clicks, search_query):
        # Ensure bar_categories has valid values or use defaults
        if not bar_categories:
            bar_categories = ['usage', 'attribute', 'performance']
            
        # Handle empty string as None
        if bar_zoom == '':
            bar_zoom = None
            
        # Get data to count categories
        categories, _, _, _, _, _ = get_bar_chart_data(
            bar_categories, bar_zoom, language, search_query if search_query else ''
        )
        
        # Set max to number of categories (minimum 10)
        return max(10, len(categories))

    @app.callback(
        [Output('y-axis-label', 'children'),
         Output('x-axis-label', 'children'),
         Output('num-y-features-label', 'children'),
         Output('num-x-features-label', 'children')],
        [Input('language-selector', 'value')]
    )
    def update_axis_labels(language):
        return (
            TRANSLATIONS[language]['y_axis_category'],
            TRANSLATIONS[language]['x_axis_category'],
            TRANSLATIONS[language]['num_y_features'],
            TRANSLATIONS[language]['num_x_features']
        )

    @app.callback(
        [Output('bar-zoom-dropdown', 'options'),
         Output('bar-zoom-dropdown', 'value')],
        [Input('bar-category-checklist', 'value'),
         Input('language-selector', 'value'),
         Input('search-button', 'n_clicks'),
         Input('bar-zoom-dropdown', 'value')],
        [State('search-input', 'value'),
         State('bar-count-slider', 'value')]
    )
    def update_bar_zoom_options(selected_categories, language, n_clicks, current_zoom, search_query, bar_count_value):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
        
        # Reset zoom when search is performed
        if trigger_id == 'search-button.n_clicks':
            return [], None
            
        search_query = search_query if search_query else ''
        
        # Default to showing 15 categories if the bar count value isn't set yet
        if bar_count_value is None or bar_count_value <= 0:
            bar_count_value = 15
            
        # If a category is already selected, check if we need to show subcategories
        if current_zoom and current_zoom != '':
            # Determine if this is a top-level category selection
            for prefix in ['U: ', 'A: ', 'P: ']:
                if current_zoom.startswith(prefix):
                    category_type = 'usage' if prefix == 'U: ' else ('attribute' if prefix == 'A: ' else 'performance')
                    selected_category = current_zoom[len(prefix):]
                    
                    # Get the dictionary mapping for the category type
                    dict_types = {
                        'usage': 'use_sents',
                        'attribute': 'attr_sents',
                        'performance': 'perf_sents'
                    }
                    
                    dict_type = dict_types[category_type]
                    
                    # Get all categories including subcategories
                    category_dict = get_cached_dict(dict_type, 'all', search_query)
            
                    # Check if this is a top-level category (no pipe character) or already a subcategory
                    is_top_level = '|' not in selected_category
                    
                    if is_top_level:
                        # Find subcategories that match the selected top-level category
                        subcategories = []
                        subcategory_counts = {}
                        total_mentions = 0
                        
                        # First pass: count total mentions and subcategory mentions
                        for path, sents_dict in category_dict.items():
                            if path.startswith(selected_category + '|'):
                                # This is a subcategory of the selected category
                                # Extract only the next level (excluding further nested levels)
                                parts = path.split('|')
                                if len(parts) > 1:
                                    # Use full hierarchy: parent|child
                                    subcategory = selected_category + '|' + parts[1]
                                    
                                    # Count mentions for this subcategory
                                    mention_count = sum(len(reviews) for sent_dict in sents_dict.values() 
                                                      for reviews in sent_dict.values())
                                    
                                    # Add to total and store count
                                    total_mentions += mention_count
                                    subcategory_counts[subcategory] = mention_count
                        
                        # Second pass: create options with percentages
                        for subcategory, count in subcategory_counts.items():
                            if total_mentions > 0:
                                percentage = (count / total_mentions) * 100
                                
                                # Use the full subcategory path with the prefix
                                display_name = prefix + subcategory
                                formatted_name = f"→ {display_name} ({percentage:.1f}%)"
                                
                                subcategories.append({
                                    'label': formatted_name,
                        'value': display_name
                    })
        
                        # If subcategories exist, display them along with the parent category and "back" option
                        if subcategories:
                            # Sort subcategories by count (descending)
                            subcategories.sort(key=lambda x: subcategory_counts[x['value'][len(prefix):]], reverse=True)
                            
                            # Add back option to return to top level
                            options = [{'label': '⬅️ ' + TRANSLATIONS[language].get('back', 'Back'), 'value': ''}]
                            
                            # Add current selection (parent category) without parentheses
                            clean_display = re.sub(r'\s*\([^)]*\)', '', current_zoom)
                            options.append({'label': f"▶ {clean_display}", 'value': current_zoom})
                            
                            # Add subcategories - also remove parentheses from display
                            for subcategory in subcategories:
                                subcategory_value = subcategory['value']
                                subcategory_label = subcategory['label']
                                # Remove any parentheses content, regardless of language or content
                                clean_subcategory = re.sub(r'\s*\([^)]*\)', '', subcategory_label)
                                options.append({'label': clean_subcategory, 'value': subcategory_value})
                            
                            return options, current_zoom
                        else:
                            # No subcategories found, just show the current category and back option
                            clean_display = re.sub(r'\s*\([^)]*\)', '', current_zoom)
                            options = [
                                {'label': '⬅️ ' + TRANSLATIONS[language].get('back', 'Back'), 'value': ''},
                                {'label': clean_display, 'value': current_zoom}
                            ]
                                                        
                            return options, current_zoom
                            
                    # If it's already a subcategory, show its parent and siblings
                    else:
                        # Extract parent category (first part before pipe)
                        parent_category = selected_category.split('|')[0]
                        full_parent = prefix + parent_category
                        
                        # Find sibling subcategories that match the parent
                        siblings = []
                        sibling_counts = {}
                        total_mentions = 0
                        
                        # First pass: count total mentions and sibling mentions
                        for path, sents_dict in category_dict.items():
                            if path.startswith(parent_category + '|'):
                                # This is a sibling subcategory
                                parts = path.split('|')
                                if len(parts) > 1:
                                    # Use full hierarchy: parent|child
                                    sibling = parent_category + '|' + parts[1]
                                    display_name = prefix + sibling
                                    
                                    # Skip the current selection
                                    if display_name == current_zoom:
                                        continue
                                    
                                    # Count mentions for this sibling
                                    mention_count = sum(len(reviews) for sent_dict in sents_dict.values() 
                                                      for reviews in sent_dict.values())
                                    
                                    # Add to total and store count
                                    total_mentions += mention_count
                                    sibling_counts[display_name] = mention_count
                        
                        # Also get count for current selection
                        current_count = 0
                        for path, sents_dict in category_dict.items():
                            if path == selected_category or path.startswith(selected_category + '|'):
                                current_count += sum(len(reviews) for sent_dict in sents_dict.values() 
                                                  for reviews in sent_dict.values())
                        total_mentions += current_count
                        
                        # Second pass: create options with percentages
                        for sibling, count in sibling_counts.items():
                            if total_mentions > 0:
                                percentage = (count / total_mentions) * 100
                                formatted_name = f"→ {sibling} ({percentage:.1f}%)"
                                
                                siblings.append({
                                    'label': formatted_name,
                                    'value': sibling
                                })
                        
                        # Sort siblings by count (descending)
                        siblings.sort(key=lambda x: sibling_counts[x['value']], reverse=True)
                        
                        # Clean parent display (no parentheses)
                        clean_parent = re.sub(r'\s*\([^)]*\)', '', full_parent)
                        
                        # Create options with back, parent (without parentheses), current selection (without parentheses) and siblings (without parentheses)
                        clean_current = re.sub(r'\s*\([^)]*\)', '', current_zoom)
                        options = [
                            {'label': '⬅️ ' + TRANSLATIONS[language].get('back', 'Back'), 'value': ''},
                            {'label': f"▶ {clean_parent}", 'value': full_parent},
                            {'label': f"▶ {clean_current} ✓", 'value': current_zoom}
                        ]
                        
                        # Add siblings - also remove parentheses from display
                        for sibling in siblings:
                            sibling_value = sibling['value']
                            sibling_label = sibling['label']
                            # Remove any parentheses content, regardless of language or content
                            clean_sibling = re.sub(r'\s*\([^)]*\)', '', sibling_label)
                            options.append({'label': clean_sibling, 'value': sibling_value})
                        
                        return options, current_zoom
                    
                    break
        
        # Default case: show top-level categories
        # Get the full list of categories that would be shown in the chart
        categories, counts, _, _, _, _ = get_bar_chart_data(
            selected_categories, None, language, search_query
        )
        
        # Apply the same limit as what would be shown in the chart
        if len(categories) > bar_count_value:
            categories = categories[:bar_count_value]
            counts = counts[:bar_count_value]
            
        # Extract just the unique top-level category names that are currently visible
        # Store them as (index, display_name) to preserve original order
        top_level_categories = []
        seen_categories = set()
        
        for i, (category, count) in enumerate(zip(categories, counts)):
            # Find the prefix (U:, A:, P:)
            for prefix in ['U: ', 'A: ', 'P: ']:
                if category.startswith(prefix):
                    # Extract the category text without the prefix
                    category_text = category[len(prefix):]
                    
                    # Check if it's a subcategory (contains a pipe)
                    if '|' in category_text:
                        # For subcategories, we group by the parent (top-level) category
                        top_level = category_text.split('|')[0].strip()
                    else:
                        # For top-level categories, just use the category itself
                        top_level = category_text
                        
                    display_name = prefix + top_level
                    
                    # Only add if not already in options
                    if display_name not in seen_categories:
                        seen_categories.add(display_name)
                        
                        # Calculate count for this top-level category - needs to include all subcategories too
                        category_count = sum(c for cat, c in zip(categories, counts) 
                                           if cat.startswith(prefix + top_level))
                        
                        # Store original index and count to maintain chart order and for percentage
                        top_level_categories.append((i, display_name, category_count))
                    break
        
        # Sort by the original index to maintain the same order as in the chart
        top_level_categories.sort(key=lambda x: x[0])
        
        # Build final options list without parentheses for parent categories
        options = []
        for _, category, count in top_level_categories:
            # Remove any existing parentheses that might be in the display name
            # This only affects display, not the actual value
            clean_display = re.sub(r'\s*\([^)]*\)', '', category)
            options.append({'label': clean_display, 'value': category})
        
        # Add "None" option at the top
        options = [{'label': '-----', 'value': ''}] + options
        
        # Keep current value if it's still valid, otherwise reset
        current_value = current_zoom
        if current_value and not any(opt['value'] == current_value for opt in options):
            current_value = None
            
        return options, current_value

    # Graph and data update callback
    @app.callback(
        [Output('x-axis-dropdown', 'options'),
        Output('y-axis-dropdown', 'options'),
        Output('x-axis-dropdown', 'value'),
        Output('y-axis-dropdown', 'value'),
        Output('correlation-matrix', 'figure'),
        Output('x-features-slider', 'max'),
        Output('y-features-slider', 'max'),
        Output('reviews-content', 'children'),
        Output('x-features-slider', 'value'),
        Output('y-features-slider', 'value'),
        Output('bar-count-slider', 'value')],
        [Input('plot-type', 'value'),
        Input('x-axis-dropdown', 'value'),
        Input('y-axis-dropdown', 'value'),
        Input('x-features-slider', 'value'),
        Input('y-features-slider', 'value'),
        Input('language-selector', 'value'),
        Input('search-button', 'n_clicks'),
        Input('bar-category-checklist', 'value'),
        Input('bar-zoom-dropdown', 'value'),
        Input('bar-count-slider', 'value')],
        [State('search-input', 'value')]
    )
    def update_graph(plot_type, x_value, y_value, top_n_x, top_n_y, language, n_clicks, 
                   bar_categories, bar_zoom, bar_count, search_query):
        
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None

        # Reset selections if search button was clicked
        if trigger_id == 'search-button.n_clicks':
            x_value = 'all'
            y_value = 'all'
        
        if x_value is None:
            x_value = 'all'
        if y_value is None:
            y_value = 'all'

        # Handle None or empty search query
        search_query = search_query if search_query else ''
        
        # Convert slider values to integers
        top_n_x = int(top_n_x)
        top_n_y = int(top_n_y)
        
        # For bar chart, we use different data and controls
        if plot_type == 'bar_chart':
            # Default values for outputs we don't use with bar chart
            x_options = []
            y_options = []
            max_x = 10
            max_y = 10
            
            # Ensure bar_categories has valid values or use defaults
            if not bar_categories:
                bar_categories = ['usage', 'attribute', 'performance']
                
            # Handle empty string as None
            if bar_zoom == '':
                bar_zoom = None
                
            # Get bar chart data
            categories, counts, sentiment_ratios, review_data, colors, title_key = get_bar_chart_data(
                bar_categories, bar_zoom, language, search_query
            )
            
            # Apply bar count limit from bar_count slider
            if bar_count > 0 and bar_count < len(categories):
                categories = categories[:bar_count]
                counts = counts[:bar_count]
                sentiment_ratios = sentiment_ratios[:bar_count]
                review_data = review_data[:bar_count]
                colors = colors[:bar_count]
            
            # Create the bar chart figure
            fig = go.Figure()
            
            # Define category type colors for legend
            type_colors = {
                'U: ': '#6A0DAD',  # Dark Purple for Usage (changed from Blue #4285F4)
                'A: ': '#34A853',  # Green for Attribute
                'P: ': '#FBBC05',  # Yellow for Performance
            }
            
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
            
            # Add an invisible satisfaction heatmap just to get the satisfaction color legend (same as matrix view)
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
            # Create a mapping of original categories to formatted (no parentheses) for click handling
            formatted_categories = []
            
            # Check if we're zoomed into a parent category
            is_zoomed = bar_zoom and bar_zoom != ''
            
            for category in categories:
                # Determine the category type prefix
                prefix = None
                for p in type_colors.keys():
                    if category.startswith(p):
                        prefix = p
                        break
                
                if prefix:
                    category_text = category[len(prefix):]
                    color = type_colors[prefix]
                    
                    # When zoomed into a parent category, we need to maintain consistent formatting with matrix plot
                    if is_zoomed and '|' in category_text:
                        # Extract the path parts
                        parts = category_text.split('|')
                        
                        # Exclude the topmost category for display (to match the example 圣诞节|圣诞树装饰)
                        if len(parts) > 1:
                            display_text = '|'.join(parts[1:])
                        else:
                            display_text = parts[0]
                          
                        # Remove any parentheses content
                        display_text = re.sub(r'\s*\([^)]*\)', '', display_text)
                    else:
                        # Standard display text for non-zoomed or parent categories
                        # Remove any parentheses content
                        display_text = prefix + category_text
                        display_text = re.sub(r'\s*\([^)]*\)', '', display_text)
                    
                    # Remember the formatted version for click handling
                    formatted_categories.append(display_text)
                    
                    ticktext.append(f'<span style="color:{color}; font-weight:bold">{display_text}</span>')
                else:
                    # Default case if no prefix matches
                    formatted_text = re.sub(r'\s*\([^)]*\)', '', category)
                    formatted_categories.append(formatted_text)
                    ticktext.append(formatted_text)
            
            # Update layout for bar chart
            fig.update_layout(
                title=TRANSLATIONS[language][title_key],
                xaxis=dict(
                    title=None,
                    tickangle=20,  # Changed from 45 to 20 to match matrix plot
                    tickmode='array',
                    tickvals=list(range(len(categories))),
                    ticktext=ticktext,
                    domain=[0, 0.85]  # Make space on the right for colorbar
                ),
                yaxis=dict(
                    title=TRANSLATIONS[language]['hover_count'],
                ),
                # Move legend to a annotations instead for better control
                showlegend=False,
                height=700,
                width=1200,
                margin=dict(
                    l=80,   # Decreased from 100
                    r=150,  # Decreased from 200
                    t=100,  # Decreased from 150
                    b=180,  # Decreased from 200
                    autoexpand=True
                ),
            )
            
            # Adjust the chart height based on number of categories
            # More categories need more height for x-axis labels
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
            
            # Create a custom legend as an annotation below the x-axis
            legend_items = []
            for prefix, color in type_colors.items():
                if any(cat.startswith(prefix) for cat in categories):
                    translated_label = TRANSLATIONS[language]['usage_category'] if prefix == 'U: ' else (
                        TRANSLATIONS[language]['attribute_category'] if prefix == 'A: ' else TRANSLATIONS[language]['performance_category']
                    )
                    legend_items.append(f'<span style="color:{color}; font-weight:bold; margin-right:15px;">▣ {prefix[0]} - {translated_label}</span>')
            
            if legend_items:
                legend_title = TRANSLATIONS[language].get('category_types', 'Category Types')
                legend_text = f'<b>{legend_title}:</b> ' + ' '.join(legend_items)
                
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=1.2,
                    y=-0.22,  # Move further down to avoid overlap with angled labels
                    text=legend_text,
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.5)",
                    borderwidth=1,
                    borderpad=6,
                    align="center"
                )
            
            # Return empty list for reviews-content when search button is clicked
            reviews_content = [] if trigger_id == 'search-button.n_clicks' else dash.no_update
            
            return (x_options, y_options, x_value, y_value, fig, max_x, max_y,
                   reviews_content, top_n_x, top_n_y, bar_count)
        else:
            # Original heatmap logic for use_attr_perf and perf_attr
            # Get data for the graph
            if plot_type == 'use_attr_perf':
                x_dict = get_cached_dict('use_sents', x_value, search_query)
                y_dict = get_cached_dict('attr_perf_ids', y_value, search_query)
            else:
                x_dict = get_cached_dict('perf_sents', x_value, search_query)
                y_dict = get_cached_dict('attr_ids', y_value, search_query)
            
            # Update slider maximums based on available features
            max_x = len(x_dict)
            max_y = len(y_dict)
            
            # For search reset, set slider values to min(7, max available)
            if trigger_id == 'search-button.n_clicks':
                top_n_x = min(7, max_x)  # Reset to default of 7 or max available
                top_n_y = min(7, max_y)  # Reset to default of 7 or max available
            else:
                # Ensure top_n values don't exceed available features
                top_n_x = min(top_n_x, max_x)
                top_n_y = min(top_n_y, max_y)
            
            matrix, sentiment_matrix, review_matrix, x_text, x_display, y_text, y_display, title_key = get_plot_data(
                plot_type, x_value, y_value, top_n_x, top_n_y, language, search_query
            )
            
            # Format display labels - remove all parentheses content for all categories
            formatted_x_display = []
            for label in x_display:
                # Remove any parentheses content, regardless of language or content
                formatted_label = re.sub(r'\s*\([^)]*\)', '', label)
                formatted_x_display.append(formatted_label)
                    
            formatted_y_display = []
            for label in y_display:
                # Remove any parentheses content, regardless of language or content
                formatted_label = re.sub(r'\s*\([^)]*\)', '', label)
                formatted_y_display.append(formatted_label)
            
            # Calculate dynamic dimensions based on number of features
            # Height calculation with increased base values for larger plot
            base_height = 700  # Increased from 500
            min_height = 600   # Increased from 400
            additional_height_per_feature = 60  # Increased from 50
            dynamic_height = max(
                min_height,
                base_height + max(0, len(y_display) - 5) * additional_height_per_feature
            )
            
            # Width calculation - increase base width for larger plot
            base_width = 1000   # Increased from 800
            min_width = 1400    # Increased from 1200
            additional_width_per_feature = 80  # Increased from 60
            max_width = 2400    # Increased from 2000
            
            # Calculate width based on the length of x-axis labels and number of features
            avg_label_length = sum(len(str(label)) for label in x_display) / len(x_display) if x_display else 0
            width_factor = max(1, avg_label_length / 15)  # Adjust width more for longer labels
            
            dynamic_width = min(
                max_width,
                max(
                    min_width,
                    base_width + max(0, len(x_display) - 5) * additional_width_per_feature * width_factor
                )
            )
            
            # Calculate dynamic margins based on label lengths - decrease margins for better space efficiency
            max_y_label_length = max(len(str(label)) for label in y_display) if y_display else 0
            left_margin = min(250, max(150, max_y_label_length * 9))  # Decreased from 350/180/10
            right_margin = 400  # Decreased from 500
            top_margin = 120    # Decreased from 180
            bottom_margin = 180 # Decreased from 250

            # Create the base heatmap figure
            fig = go.Figure()

            # Reorder x-axis by frequency (sum of mentions for each category)
            if len(matrix) > 0 and len(matrix[0]) > 0:
                # Calculate total mentions for each x-axis category
                x_totals = np.sum(matrix, axis=0)
                
                # Get indices sorted by frequency (descending)
                x_sorted_indices = np.argsort(-x_totals)
                
                # Reorder all x-related data
                matrix = matrix[:, x_sorted_indices]
                sentiment_matrix = sentiment_matrix[:, x_sorted_indices]
                review_matrix = review_matrix[:, x_sorted_indices]
                x_text = [x_text[i] for i in x_sorted_indices]
                x_display = [x_display[i] for i in x_sorted_indices]

            # Format display labels - remove all parentheses content for all categories
            formatted_x_display = []
            for label in x_display:
                # Remove any parentheses content, regardless of language or content
                formatted_label = re.sub(r'\s*\([^)]*\)', '', label)
                formatted_x_display.append(formatted_label)
                    
            formatted_y_display = []
            for label in y_display:
                # Remove any parentheses content, regardless of language or content
                formatted_label = re.sub(r'\s*\([^)]*\)', '', label)
                formatted_y_display.append(formatted_label)

            # Add the satisfaction ratio heatmap (color legend)
            fig.add_trace(go.Heatmap(
                z=sentiment_matrix,
                x=formatted_x_display,  # Use formatted display labels
                y=formatted_y_display,  # Use formatted display labels
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

            # Get width legend translations
            width_legend_text = get_width_legend_translations(language)['explanation']

            # Add the custom shapes for the actual visualization
            max_mentions = np.max(matrix)
            min_mentions = np.min(matrix[matrix > 0]) if np.any(matrix > 0) else 0  # Get minimum non-zero value
            
            # Add the shapes using log width
            for i in range(len(y_display)):
                for j in range(len(x_display)):
                    if matrix[i, j] > 0:
                        box_width = get_log_width(matrix[i, j], max_mentions, min_mentions)
                        fig.add_shape(
                            type="rect",
                            x0=j-box_width/2, y0=i-0.4, x1=j+box_width/2, y1=i+0.4,
                            fillcolor=ratio_to_rgb(sentiment_matrix[i, j]),
                            line_color="rgba(0,0,0,0)",
                        )

            # Find a good example box from the actual data
            if np.any(matrix > 0):
                example_i, example_j = np.unravel_index(np.argmax(matrix), matrix.shape)
                example_sentiment = sentiment_matrix[example_i, example_j]
                example_count = int(matrix[example_i, example_j])
            else:
                example_sentiment = 0.5
            
            # Constants for width example box and explanation positioning
            EXAMPLE_BOX_LEFT = 1.05
            EXAMPLE_BOX_RIGHT = EXAMPLE_BOX_LEFT + 0.12
            EXAMPLE_BOX_BOTTOM = 1 
            EXAMPLE_BOX_TOP = EXAMPLE_BOX_BOTTOM + 0.11

            BRACKET_TOP = EXAMPLE_BOX_TOP + 0.06
            BRACKET_VERTICAL_LENGTH = 0.03  # Length of vertical bracket lines

            EXPLANATION_X = EXAMPLE_BOX_LEFT + (EXAMPLE_BOX_RIGHT - EXAMPLE_BOX_LEFT) * 1.5 # Center explanation above bracket
            EXPLANATION_Y = BRACKET_TOP + 0.1  # Position explanation above bracket

            # Add explanation text with translation above the bracket - adjusted position
            fig.add_annotation(
                xref="paper", yref="paper", 
                x=EXPLANATION_X,
                y=EXPLANATION_Y,
                text=width_legend_text,
                showarrow=False,
                font=dict(size=13),
                align="right",
                width=220
            )
            
            # Add the color scale legend title with larger font
            fig.update_coloraxes(
                colorbar=dict(
                    title=dict(
                        text=TRANSLATIONS[language]['satisfaction_level'],
                        font=dict(size=14)  # Increased font size
                    ),
                    tickfont=dict(size=12)  # Increased tick font size
                )
            )

            # Add the example box
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
            else:  # perf_attr
                x_category_type = AXIS_CATEGORY_NAMES[language]['perf']
                y_category_type = AXIS_CATEGORY_NAMES[language]['attr']

            fig.update_layout(
                title=TRANSLATIONS[language][title_key],
                xaxis=dict(
                    tickangle=20,
                    tickmode='array',
                    tickvals=list(range(len(formatted_x_display))),
                    ticktext=formatted_x_display,
                    tickfont=dict(size=12),  # Increased from 10
                    title=dict(
                        text=f"{TRANSLATIONS[language]['x_axis_percentage']} {x_category_type}",
                        font=dict(size=14)   # Increased from 12
                    )
                ),
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(formatted_y_display))),
                    ticktext=formatted_y_display,
                    tickfont=dict(size=12),  # Increased from 10
                    title=dict(
                        text=f"{y_category_type} {TRANSLATIONS[language]['y_axis_percentage']}",
                        font=dict(size=14)   # Increased from 12
                    )
                ),
                font=dict(size=13),  # Global font size increase
                title_font=dict(size=18),  # Title font size increase
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

            # Get hover translations
            hover_translations = get_hover_translations(language)

            # Update hover template with translations
            hover_template = (
                f"{hover_translations['hover_x']}: %{{x}}<br>" +
                f"{hover_translations['hover_y']}: %{{y}}<br>" +
                f"{hover_translations['hover_count']}: %{{text}}<br>" +
                f"{hover_translations['hover_satisfaction']}: %{{customdata:.2f}}"
            )

            fig.update_traces(
                hovertemplate=hover_template,
                customdata=sentiment_matrix,
                text=matrix
            )

            x_options = get_options(x_value, x_text)  # Use clean text for options
            y_options = get_options(y_value, y_text[::-1])  # Use clean text for options, reverse order
            
            # Return empty list for reviews-content when search button is clicked
            reviews_content = [] if trigger_id == 'search-button.n_clicks' else dash.no_update
            
            # Return both the new slider values and the rest of the outputs
            return (x_options, y_options, x_value, y_value, fig, max_x, max_y, 
                    reviews_content, top_n_x, top_n_y, dash.no_update)

    # Reviews update callback
    @app.callback(
        [Output('reviews-content', 'children', allow_duplicate=True),
        Output('sentiment-filter', 'value'),
        Output('sentiment-filter', 'style'),
        Output('review-counts', 'children')],
        [Input('sentiment-filter', 'value'),
        Input('correlation-matrix', 'clickData'),
        Input('language-selector', 'value')],
        [State('plot-type', 'value'),
        State('x-axis-dropdown', 'value'),
        State('y-axis-dropdown', 'value'),
        State('x-features-slider', 'value'),
        State('y-features-slider', 'value'),
        State('search-input', 'value'),
        State('bar-category-checklist', 'value'),
        State('bar-zoom-dropdown', 'value'),
        State('bar-count-slider', 'value')],
        prevent_initial_call=True
    )
    def update_reviews_with_sentiment_filter(sentiment_filter, click_data, language, 
                                            plot_type, x_value, y_value, top_n_x, top_n_y, 
                                            search_query, bar_categories, bar_zoom, bar_count):
        from dash import html
        
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
        
        if trigger_id == 'search-button.n_clicks':
            return [], 'show_all', {'display': 'none'}, ''
            
        if trigger_id == 'correlation-matrix.clickData':
            sentiment_filter = 'show_all'
        
        if click_data:
            search_query = search_query if search_query else ''
            
            if plot_type == 'bar_chart':
                # Get clicked category from the bar chart
                point = click_data['points'][0]
                clicked_category = point['x']
                
                # Get bar chart data
                if not bar_categories:
                    bar_categories = ['usage', 'attribute', 'performance']
                
                # Handle empty string as None
                if bar_zoom == '':
                    bar_zoom = None
                    
                categories, counts, sentiment_ratios, review_data, colors, title_key = get_bar_chart_data(
                    bar_categories, bar_zoom, language, search_query
                )
                
                # Apply bar count limit from bar_count slider
                if bar_count > 0 and bar_count < len(categories):
                    categories = categories[:bar_count]
                    counts = counts[:bar_count]
                    sentiment_ratios = sentiment_ratios[:bar_count]
                    review_data = review_data[:bar_count]
                    colors = colors[:bar_count]
                
                # Create formatted category names to match what's displayed in the chart
                formatted_to_original = {}
                
                # Check if we're zoomed into a parent category
                is_zoomed = bar_zoom and bar_zoom != ''
                
                # Define category type colors for mapping
                type_colors = {
                    'U: ': '#6A0DAD',  # Dark Purple for Usage (changed from Blue #4285F4)
                    'A: ': '#34A853',  # Green for Attribute
                    'P: ': '#FBBC05',  # Yellow for Performance
                }
                
                for i, category in enumerate(categories):
                    # Determine the category type prefix
                    prefix = None
                    for p in type_colors.keys():
                        if category.startswith(p):
                            prefix = p
                            break
                            
                    if prefix:
                        category_text = category[len(prefix):]
                        
                        # When zoomed into a parent category, format display matches matrix plot
                        if is_zoomed and '|' in category_text:
                            # Extract the path parts
                            parts = category_text.split('|')
                            
                            # Exclude the topmost category for display (same format as in the graph)
                            if len(parts) > 1:
                                display_text = '|'.join(parts[1:])
                            else:
                                display_text = parts[0]
                            
                            # Remove any parentheses content
                            display_text = re.sub(r'\s*\([^)]*\)', '', display_text)
                            
                            # Map the formatted display text to its index
                            formatted_to_original[display_text] = i
                            
                            # Also map individual parts for fallback - users might click on subpart
                            for part in parts[1:]:  # Only map sub-parts, not the topmost category
                                part = part.strip()
                                if part:
                                    formatted_to_original[part] = i
                                    
                            # Also map the full format without parentheses as fallback
                            full_formatted = re.sub(r'\s*\([^)]*\)', '', category)
                            formatted_to_original[full_formatted] = i
                        else:
                            # Standard format without parentheses
                            display_text = prefix + category_text
                            formatted = re.sub(r'\s*\([^)]*\)', '', display_text)
                            formatted_to_original[formatted] = i
                    else:
                        # Default case
                        formatted = re.sub(r'\s*\([^)]*\)', '', category)
                        formatted_to_original[formatted] = i
                
                try:
                    # Find the index of clicked category in results using the formatted version
                    idx = formatted_to_original.get(clicked_category)
                    if idx is None:
                        # If not found by formatted name, try direct match (backward compatibility)
                        idx = categories.index(clicked_category)
                        
                    reviews = review_data[idx]
                    
                    # Count positive and negative reviews
                    positive_reviews = []
                    negative_reviews = []
                    neutral_reviews = []
                    for review in reviews:
                        if f'<span style="background-color: {color_mapping["+"]}' in review:
                            positive_reviews.append(review)
                        elif f'<span style="background-color: {color_mapping["-"]}' in review:
                            negative_reviews.append(review)
                        else:
                            neutral_reviews.append(review)
                    
                    # Filter reviews based on selection
                    if sentiment_filter == 'show_positive':
                        filtered_reviews = positive_reviews
                    elif sentiment_filter == 'show_negative':
                        filtered_reviews = negative_reviews
                    else:  # 'all'
                        filtered_reviews = reviews
                    
                    # Get highlight examples for current language
                    x_highlight, y_highlight, pos_highlight, neg_highlight = get_highlight_examples(language)
                    
                    # For display in the header, we use the original category (with percentages)
                    # to provide more information to the user
                    original_category = categories[idx]
                    category_clicked = f'<span style="background-color: {colors[idx]}; color: white; padding: 5px; border-radius: 3px;">{original_category}</span>'
                    
                    # Create header HTML with sentiment filter
                    review_format = TRANSLATIONS[language]['review_format']
                    
                    header_html = f"""
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
                        <div>{TRANSLATIONS[language]['selected']} {category_clicked}</div>
                        <div style="margin-top: 3px;">
                            {review_format}
                        </div>
                    </body>
                    </html>
                    """
                    
                    content = [
                        html.Iframe(
                            srcDoc=header_html,
                            style={
                                'width': '100%',
                                'height': '120px',
                                'border': 'none',
                                'backgroundColor': 'white'
                            }
                        )
                    ]
                    
                    if filtered_reviews:
                        reviews_html = []
                        for idx, review in enumerate(filtered_reviews):
                            reviews_html.append(f"""
                                <div class="review-container">
                                    <span>{TRANSLATIONS[language]['review']} {idx + 1}: </span>
                                    {review}
                                </div>
                            """)
                        
                        html_content = f"""
                        <html>
                        <head>
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
                                }}
                                .review-container:last-child {{
                                    border-bottom: none;
                                }}
                            </style>
                        </head>
                        <body>
                            {''.join(reviews_html)}
                        </body>
                        </html>
                        """
                        
                        content.append(html.Iframe(
                            srcDoc=html_content,
                            style={
                                'width': '100%',
                                'height': '630px',
                                'border': 'none',
                                'borderRadius': '3px',
                                'backgroundColor': 'white'
                            }
                        ))
                    else:
                        content.append(html.P(TRANSLATIONS[language]['no_reviews']))
                    
                    # Create count display based on filter selection
                    count_display = []
                    if sentiment_filter == 'show_all':
                        count_display = [
                            html.Span(f"{TRANSLATIONS[language]['positive_count'].format(len(positive_reviews))} | "),
                            html.Span(TRANSLATIONS[language]['negative_count'].format(len(negative_reviews)))
                        ]
                    elif sentiment_filter == 'show_positive':
                        count_display = [html.Span(TRANSLATIONS[language]['positive_count'].format(len(positive_reviews)))]
                    else:  # show_negative
                        count_display = [html.Span(TRANSLATIONS[language]['negative_count'].format(len(negative_reviews)))]
                    
                    # Update filter style to show horizontally
                    filter_style = {
                        'display': 'block',
                        'marginBottom': '5px',  # Decreased from 10px
                        'whiteSpace': 'nowrap',  # Prevent wrapping
                        'display': 'flex',
                        'flexDirection': 'row',
                        'gap': '15px',  # Decreased from 20px
                        'alignItems': 'center'
                    }
                    
                    return content, sentiment_filter, filter_style, count_display
                    
                except (ValueError, IndexError) as e:
                    print(f"Error processing bar chart click data: {str(e)}")
                    return dash.no_update, sentiment_filter, dash.no_update, dash.no_update
            else:
                # Original heatmap click handling
                point = click_data['points'][0]
                clicked_x = point['x']
                clicked_y = point['y']
                
                matrix, sentiment_matrix, review_matrix, x_text, x_display, y_text, y_display, title_key = get_plot_data(
                    plot_type, x_value, y_value, top_n_x, top_n_y, language, search_query
                )
                
                # Format display labels - remove all parentheses content for all categories
                formatted_x_display = []
                for label in x_display:
                    # Remove any parentheses content, regardless of language or content
                    formatted_label = re.sub(r'\s*\([^)]*\)', '', label)
                    formatted_x_display.append(formatted_label)
                    
                formatted_y_display = []
                for label in y_display:
                    # Remove any parentheses content, regardless of language or content
                    formatted_label = re.sub(r'\s*\([^)]*\)', '', label)
                    formatted_y_display.append(formatted_label)
                
                # Create a mapping from formatted labels to original indices
                x_mapping = {formatted: i for i, formatted in enumerate(formatted_x_display)}
                y_mapping = {formatted: i for i, formatted in enumerate(formatted_y_display)}
                
                try:
                    # Find the indices using the mapping with formatted labels
                    i = y_mapping.get(clicked_y)
                    j = x_mapping.get(clicked_x)
                    
                    # If not found by direct mapping, try a more lenient search
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
                        
                        # Count positive and negative reviews - moved outside the sentiment filter condition
                        positive_reviews = []
                        negative_reviews = []
                        neutral_reviews = []
                        for review in reviews:
                            if f'<span style="background-color: {color_mapping["+"]}' in review:
                                positive_reviews.append(review)
                            elif f'<span style="background-color: {color_mapping["-"]}' in review:
                                negative_reviews.append(review)
                            else:
                                neutral_reviews.append(review)
                        
                        # Filter reviews based on selection
                        if sentiment_filter == 'show_positive':
                            filtered_reviews = positive_reviews
                        elif sentiment_filter == 'show_negative':
                            filtered_reviews = negative_reviews
                        else:  # 'all'
                            filtered_reviews = reviews
                        
                        # Get highlight examples for current language
                        x_highlight, y_highlight, pos_highlight, neg_highlight = get_highlight_examples(language)
                        
                        # For the header display, use the original values with percentages
                        # to provide more information to the user
                        original_x = x_display[j]
                        original_y = y_display[i]
                        
                        x_clicked = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["x"]}; color: black; padding: 5px;">X=<b>{original_x}</b></span>'
                        y_clicked = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["y"]}; color: black; padding: 5px;">Y=<b>{original_y}</b></span>'
                        
                        # Create header HTML with sentiment filter
                        review_format = TRANSLATIONS[language]['review_format']
                        
                        header_html = f"""
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
                            <div>{TRANSLATIONS[language]['selected']} {x_clicked}, {y_clicked}</div>
                            <div style="margin-top: 3px;">
                                {review_format}
                            </div>
                        </body>
                        </html>
                        """

                        content = [
                            html.Iframe(
                                srcDoc=header_html,
                                style={
                                    'width': '100%',
                                    'height': '120px',
                                    'border': 'none',
                                    'backgroundColor': 'white'
                                }
                            )
                        ]
                        
                        if filtered_reviews:
                            reviews_html = []
                            for idx, review in enumerate(filtered_reviews):
                                reviews_html.append(f"""
                                    <div class="review-container">
                                        <span>{TRANSLATIONS[language]['review']} {idx + 1}: </span>
                                        {review}
                                    </div>
                                """)
                            
                            html_content = f"""
                            <html>
                            <head>
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
                                    }}
                                    .review-container:last-child {{
                                        border-bottom: none;
                                    }}
                                </style>
                            </head>
                            <body>
                                {''.join(reviews_html)}
                            </body>
                            </html>
                            """
                            
                            content.append(html.Iframe(
                                srcDoc=html_content,
                                style={
                                    'width': '100%',
                                    'height': '630px',
                                    'border': 'none',
                                    'borderRadius': '3px',
                                    'backgroundColor': 'white'
                                }
                            ))
                        else:
                            content.append(html.P(TRANSLATIONS[language]['no_reviews']))
                        
                        # Create count display based on filter selection
                        count_display = []
                        if sentiment_filter == 'show_all':
                            count_display = [
                                html.Span(f"{TRANSLATIONS[language]['positive_count'].format(len(positive_reviews))} | "),
                                html.Span(TRANSLATIONS[language]['negative_count'].format(len(negative_reviews)))
                            ]
                        elif sentiment_filter == 'show_positive':
                            count_display = [html.Span(TRANSLATIONS[language]['positive_count'].format(len(positive_reviews)))]
                        else:  # show_negative
                            count_display = [html.Span(TRANSLATIONS[language]['negative_count'].format(len(negative_reviews)))]
                        
                        # Update filter style to show horizontally
                        filter_style = {
                            'display': 'block',
                            'marginBottom': '5px',  # Decreased from 10px
                            'whiteSpace': 'nowrap',  # Prevent wrapping
                            'display': 'flex',
                            'flexDirection': 'row',
                            'gap': '15px',  # Decreased from 20px
                            'alignItems': 'center'
                        }
                        
                        return content, sentiment_filter, filter_style, count_display
                        
                except ValueError as e:
                    print(f"Error processing click data: {str(e)}")
                    
        return dash.no_update, sentiment_filter, dash.no_update, dash.no_update

    # Search-related callbacks
    @app.callback(
        Output('search-examples', 'children'),
        [Input('language-selector', 'value')]
    )
    def update_search_examples(language):
        return get_search_examples_html(language)

    @app.callback(
        Output('search-results-info', 'children'),
        [Input('search-button', 'n_clicks'),
        Input('language-selector', 'value')],
        [State('search-input', 'value'),
        State('plot-type', 'value'),
        State('x-axis-dropdown', 'value'),
        State('y-axis-dropdown', 'value')]
    )
    def update_search_results_info(n_clicks, language, search_query, plot_type, x_value, y_value):
        if not search_query or not search_query.strip():
            return ''
            
        # Get the dictionaries based on plot type
        if plot_type == 'use_attr_perf':
            x_dict = get_cached_dict('use_sents', x_value, search_query)
            y_dict = get_cached_dict('attr_perf_sents', y_value, search_query)
        else:
            x_dict = get_cached_dict('perf_sents', x_value, search_query)
            y_dict = get_cached_dict('attr_sents', y_value, search_query)
        
        # Count total unique reviews
        total_reviews = set()
        
        # Count reviews in x_dict
        for sents_dict in x_dict.values():
            for sent_dict in sents_dict.values():
                if isinstance(sent_dict, dict):
                    for reviews in sent_dict.values():
                        total_reviews.update(reviews)
                elif isinstance(sent_dict, list):
                    total_reviews.update(sent_dict)
                    
        # Count reviews in y_dict
        for sents_dict in y_dict.values():
            for sent_dict in sents_dict.values():
                if isinstance(sent_dict, dict):
                    for reviews in sent_dict.values():
                        total_reviews.update(reviews)
                elif isinstance(sent_dict, list):
                    total_reviews.update(sent_dict)
        
        # Normalize search query
        normalized_query = normalize_search_query(search_query)
        
        # Return translated message with results
        return TRANSLATIONS[language]['search_results'].format(len(total_reviews), normalized_query)

    @app.callback(
        Output('search-input', 'placeholder'),
        [Input('language-selector', 'value')]
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
        Output('sentiment-filter', 'options'),
        [Input('language-selector', 'value')]
    )
    def update_sentiment_filter_options(language):
        return [
            {'label': TRANSLATIONS[language]['show_all'], 'value': 'show_all'},
            {'label': TRANSLATIONS[language]['show_positive'], 'value': 'show_positive'},
            {'label': TRANSLATIONS[language]['show_negative'], 'value': 'show_negative'}
        ] 