import dash
from dash import Input, Output, State, html, dcc
from flask_login import login_user, logout_user, UserMixin, current_user
import json
from urllib.parse import parse_qs
import logging
from datetime import datetime

from config import TRANSLATIONS, VALID_USERNAME, VALID_PASSWORD
from data import get_cached_dict, get_bar_chart_data, update_raw_dict_map, get_plot_data
from trend_chart import update_trend_chart_data, get_trend_line_options, get_time_bucket_options
from bar_chart import process_bar_chart_data
from matrix_chart import process_matrix_data
from utils import format_category_display
from review_display import (
    create_count_display, get_filter_style, update_reviews_with_filters,
    create_review_display, categorize_reviews, filter_reviews_by_sentiment
)
from date_filter import handle_date_filter, update_date_from_input, update_date_display
from search import (
    get_search_examples_html,
    logic_toggle_search_help,
    logic_update_search_results_info
)
from logging_config import log_user_action as log_user, log_error, log_app_event

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

def log_user_action(real_name, pathname, action, context_dict):
    """Helper function to log user actions in JSON format."""
    try:
        # Use the new logging system
        log_user(real_name, pathname, action, context_dict)
    except Exception as e:
        log_error(f"Failed to log user action: {action}", exc_info=True)

def register_callbacks(app):
    """Register all callbacks with the app."""
    
    @app.callback(
        [Output('url', 'pathname'),
         Output('url', 'search'),
         Output('login-error', 'children'),
         Output('user-real-name-state', 'data')], # Changed children to data
        [Input('login-button', 'n_clicks')],
        [State('username-input', 'value'),
         State('password-input', 'value'),
         State('real-name-input', 'value'), # Added state for real name input
         State('app-language-state', 'children'),
         State('category-state', 'children')],
        prevent_initial_call=True
    )
    def login_callback(n_clicks, username, password, real_name, language, category): # Added real_name
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
            
        # Check if username and password are None or empty
        if not username or not password:
            return '/login', f'?lang={language}&category={category}', TRANSLATIONS[language]['invalid_credentials'], dash.no_update # No update for name
            
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            user = User(username)
            login_success = login_user(user)
            
            # Log successful login
            log_user_action(real_name, '/login', 'login_success', {'username': username})
            
            # Return to home page with language and category parameters preserved, store real name
            return '/', f'?lang={language}&category={category}', '', real_name # Return real_name
        
        # Log failed login attempt
        log_user_action(real_name, '/login', 'login_fail', {'username': username})
        # Return to login page with error and parameters
        return '/login', f'?lang={language}&category={category}', TRANSLATIONS[language]['invalid_credentials'], dash.no_update # No update for name

    @app.callback(
        [Output('url', 'pathname', allow_duplicate=True),
         Output('url', 'search', allow_duplicate=True)],
        [Input('logout-button', 'n_clicks')],
        [State('url', 'pathname'),
         State('app-language-state', 'children'),
         State('category-state', 'children'),
         State('user-real-name-state', 'data')], # Added real name state
        prevent_initial_call=True
    )
    def logout_callback(n_clicks, pathname, language, category, real_name): # Added real_name
        if n_clicks:
            # Log logout action before logging out
            log_user_action(real_name, pathname, 'logout', {})
            logout_user()
            # Redirect to login page with parameters
            return '/login', f'?lang={language}&category={category}'
        # Default case - stay on current page
        return pathname, f'?lang={language}&category={category}'

    # Language-dependent callbacks
    @app.callback(
        Output('plot-type', 'children'),
        [Input('app-language-state', 'children')]
    )
    def update_plot_type_labels(language):
        # Create a list of dcc.Tab components with translated labels
        tabs = [
            dcc.Tab(label=TRANSLATIONS[language]['bar_chart'], value='bar_chart'),
            dcc.Tab(label=TRANSLATIONS[language]['use_vs_attr_perf'], value='use_attr_perf'),
            dcc.Tab(label=TRANSLATIONS[language]['perf_vs_attr'], value='perf_attr')
        ]
        # Return the list of tabs for the 'children' property
        return tabs

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
        return min(max(len(display_categories), 1), 30)

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
         Output('total-reviews-count', 'data'),
         Output('selected-words-storage', 'children'),  # Add output to reset word selection
         Output('sentiment-filter', 'value', allow_duplicate=True)],  # Add allow_duplicate=True
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
         State('date-filter-storage', 'children'),
         State('user-real-name-state', 'data')], # Add user real name state
        prevent_initial_call=True
    )
    def update_visualization_and_controls(plot_type, x_value, y_value, top_n_x, top_n_y, language, n_clicks, 
                   bar_categories, bar_zoom_value, bar_count, date_slider_value, pathname, search, search_query, date_filter_storage, user_real_name): # Add user_real_name
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
        triggered_by_user = trigger_id and trigger_id not in ['url.pathname', 'url.search', 'app-language-state.children']
        
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
        
        # Clear reviews if the plot type changes, or if zoom/axis settings change
        should_clear_reviews = (
            trigger_id == 'plot-type.value' or 
            trigger_id == 'bar-zoom-dropdown.value' or
            trigger_id == 'x-axis-dropdown.value' or
            trigger_id == 'y-axis-dropdown.value' or
            trigger_id == 'x-features-slider.value' or
            trigger_id == 'y-features-slider.value'
        )
        
        if should_clear_reviews:
            reviews_content = []
            
        # Handle special case for plot type changes - log and reset reviews
        if trigger_id == 'plot-type.value':
            log_user_action(user_real_name, pathname, 'change_plot_type', {'plot_type': plot_type})
        
        # Reset for sentiment filter and word selection when changing visualization settings
        reset_sentiment_filter = 'show_all' if should_clear_reviews else dash.no_update
        reset_word_selection = '[]' if should_clear_reviews else dash.no_update
        
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
            log_user_action(user_real_name, pathname, 'search', {'query': search_query, 'start_date': start_date, 'end_date': end_date})
        
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
        
        # Log other user-triggered updates
        if triggered_by_user and trigger_id != 'plot-type.value' and trigger_id != 'search-button.n_clicks':
             log_context = {
                 'trigger_id': trigger_id,
                 'plot_type': plot_type,
                 'language': language,
                 'search_query': search_query,
                 'start_date': start_date,
                 'end_date': end_date,
             }
             if plot_type == 'bar_chart':
                 log_context.update({
                     'bar_categories': bar_categories,
                     'bar_zoom': bar_zoom_value,
                     'bar_count': bar_count
                 })
             else:
                 log_context.update({
                     'x_axis': x_value,
                     'y_axis': y_value,
                     'top_n_x': top_n_x,
                     'top_n_y': top_n_y
                 })
             log_user_action(user_real_name, pathname, 'update_visualization', log_context)

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
                30,                     # x-features-slider max
                30,                     # y-features-slider max
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
                total_reviews if should_update_count else dash.no_update,  # total-reviews-count
                reset_word_selection,   # reset selected words
                reset_sentiment_filter  # reset sentiment filter
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
                min(max(max_x, 1), 30),                  # x-features-slider max
                min(max(max_y, 1), 30),                  # y-features-slider max
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
                total_reviews if should_update_count else dash.no_update,  # total-reviews-count
                reset_word_selection,   # reset selected words
                reset_sentiment_filter  # reset sentiment filter
            )

    # Reviews update callback - modify to reset filters on click
    @app.callback(
        [Output('reviews-content', 'children', allow_duplicate=True),
         Output('sentiment-filter', 'value', allow_duplicate=True),
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
         State('selected-words-storage', 'children'),
         State('user-real-name-state', 'data'), # Added state
         State('url', 'pathname')], # Added state
        prevent_initial_call=True
    )
    def update_reviews_with_sentiment_filter(sentiment_filter, click_data, language, 
                                        plot_type, x_value, y_value, top_n_x, top_n_y, 
                                        search_query, bar_categories, bar_zoom, bar_count, 
                                        date_filter_storage, selected_words_json, user_real_name, pathname): # Added states
        # --- Get User Info for Logging ---
        # Unfortunately, we can't easily get user_real_name here without adding another State,
        # which might overly complicate this callback. We'll log without it for now,
        # or assume it can be correlated later via timestamp/session.
        # A better approach might involve storing real_name in the session/user object directly.
        # --- End User Info ---

        # Parse date range from storage
        date_range = json.loads(date_filter_storage) if date_filter_storage else {"start_date": None, "end_date": None}
        start_date = date_range.get("start_date")
        end_date = date_range.get("end_date")
        
        # --- Initialize Outputs to Safe Defaults ---
        reviews_content_output = html.Div([html.Div(TRANSLATIONS[language]['no_reviews'], style={'padding': '20px', 'textAlign': 'center'})])
        positive_reviews_output = None
        negative_reviews_output = None
        sentiment_filter_output = sentiment_filter # Keep incoming filter value initially
        filter_style_output = {'display': 'none'} # Default to hidden
        count_display_output = '' # Default to empty
        selected_words = []  # Always reset selected words on a new click
        selected_words_output = '[]'  # Always reset word selection on a new click

        # --- Determine Trigger and Setup Logging ---
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
        log_action = None
        log_context = {}
        if trigger_id and 'main-figure.clickData' in trigger_id:
            if click_data:
                # Reset sentiment filter on any new click
                sentiment_filter_output = 'show_all'
                
                log_action = 'select_graph_element'
                log_context = {
                    'click_data': str(click_data),
                    'plot_type': plot_type,
                    'current_sentiment_filter': sentiment_filter,
                    'language': language,
                    'search_query': search_query,
                    'start_date': start_date,
                    'end_date': end_date,
                }
                if plot_type == 'bar_chart':
                     log_context.update({'bar_categories': bar_categories, 'bar_zoom': bar_zoom, 'bar_count': bar_count})
                else:
                     log_context.update({'x_axis': x_value, 'y_axis': y_value, 'top_n_x': top_n_x, 'top_n_y': top_n_y})
            else:
                 raise dash.exceptions.PreventUpdate # Invalid click data

        elif trigger_id and 'sentiment-filter.value' in trigger_id:
            log_action = 'filter_reviews_by_sentiment'
            log_context = {
                'click_data_source': str(click_data),
                'new_sentiment_filter': sentiment_filter,
                'selected_words': selected_words,
                'language': language,
                'search_query': search_query,
                'start_date': start_date,
                'end_date': end_date
            }
            # Don't reset words when only changing sentiment filter
            selected_words = json.loads(selected_words_json) if selected_words_json else []
            selected_words_output = selected_words_json  # Keep existing word selection
        # --- End Logging Setup ---

        # --- Determine if Processing is Needed ---
        process_click = trigger_id and 'main-figure.clickData' in trigger_id and click_data
        process_sentiment_change = trigger_id and 'sentiment-filter.value' in trigger_id

        # --- Early Exits for Non-User Actions ---
        if not process_click and not process_sentiment_change and not (trigger_id and 'app-language-state.children' in trigger_id):
             return reviews_content_output, sentiment_filter_output, filter_style_output, count_display_output, selected_words_output
        if (trigger_id and 'app-language-state.children' in trigger_id) and not process_click and not process_sentiment_change:
             return reviews_content_output, sentiment_filter_output, filter_style_output, count_display_output, selected_words_output
        # --- End Early Exits ---

        # --- Main Processing Block ---
        if process_click or process_sentiment_change:
            try:
                # --- Bar Chart Logic ---
                if plot_type == 'bar_chart':
                    if not bar_categories: raise ValueError("Bar categories missing.")
                    current_bar_zoom = bar_zoom if bar_zoom and bar_zoom != 'all' else 'all'
                    # Fetch the *full* data corresponding to the current filters, *before* bar_count truncation
                    _, original_categories_all, _, _, review_data_all, _, _ = get_bar_chart_data(
                        bar_categories, current_bar_zoom, language, search_query, start_date, end_date
                    )
                    # Also get the currently displayed categories for context (truncated)
                    display_categories_displayed = []
                    if bar_count > 0 and bar_count <= len(original_categories_all):
                         display_categories, _, _, _, _, _, _ = get_bar_chart_data(
                             bar_categories, current_bar_zoom, language, search_query, start_date, end_date
                         )
                         display_categories_displayed = display_categories[:bar_count]


                    if click_data and 'points' in click_data and len(click_data['points']) > 0 and 'customdata' in click_data['points'][0]:
                        clicked_custom_data = click_data['points'][0]['customdata']
                        if isinstance(clicked_custom_data, list) and len(clicked_custom_data) >= 3:
                            # customdata = [count, sentiment_ratio, original_category, display_category]
                            clicked_original_category = clicked_custom_data[2]  # Get original category at index 2
                            
                            try:
                                # Find the index in the *untruncated* list
                                idx_in_all = original_categories_all.index(clicked_original_category)
                                review_dicts = review_data_all[idx_in_all]

                                # Extract review_infos
                                if review_dicts and isinstance(review_dicts[0], dict) and 'review_info' in review_dicts[0]:
                                    review_infos = [item['review_info'] for item in review_dicts if 'review_info' in item]
                                else:
                                    review_infos = review_dicts # Assuming review_dicts is already the list of review infos

                                # Get positive/negative reviews
                                positive_reviews, negative_reviews, _ = categorize_reviews(review_infos)
                                filtered_reviews_for_display = filter_reviews_by_sentiment(review_infos, positive_reviews, negative_reviews, sentiment_filter_output)

                                # Use display category from customdata if available (index 3), otherwise use original category
                                clicked_display_category = clicked_custom_data[3] if len(clicked_custom_data) >= 4 else clicked_original_category
                                
                                # If we still need to find a formatted display category from the available categories
                                if display_categories_displayed:
                                    try:
                                        # Find index in the full list to map to the truncated display list
                                        display_idx = original_categories_all.index(clicked_original_category)
                                        if display_idx < len(display_categories_displayed):
                                            clicked_display_category = display_categories_displayed[display_idx]
                                        # else: The clicked category might not be in the truncated display list if bar_count changed
                                    except ValueError:
                                        # Fallback to the direct display category if we can't map it
                                        pass

                                # --- Update Outputs on Success --- #
                                positive_reviews_output = positive_reviews
                                negative_reviews_output = negative_reviews
                                reviews_content_output = create_review_display(
                                    filtered_reviews_for_display, language, plot_type='bar_chart',
                                    x_category=clicked_display_category, selected_words=selected_words
                                )
                                # filter_style_output = get_filter_style() # Set later based on success
                            except (ValueError, IndexError) as e:
                                # Keep outputs as None/default
                                pass
                        else:
                            # Keep outputs as None/default
                            pass

                # --- Matrix Logic ---
                elif plot_type in ['use_attr_perf', 'perf_attr']:
                    if click_data and 'points' in click_data and len(click_data['points']) > 0:
                        point = click_data['points'][0]
                        clicked_x = point['x']
                        clicked_y = point['y']
                        matrix, _, review_matrix, x_text, _, y_text, _, _ = get_plot_data(
                            plot_type, x_value, y_value, top_n_x, top_n_y, language, search_query, start_date, end_date
                        )
                        formatted_x_display = [format_category_display(label, language) for label in x_text]
                        formatted_y_display = [format_category_display(label, language) for label in y_text]
                        x_mapping = {formatted: i for i, formatted in enumerate(formatted_x_display)}
                        y_mapping = {formatted: i for i, formatted in enumerate(formatted_y_display)}
                        try:
                            i = y_mapping.get(clicked_y)
                            j = x_mapping.get(clicked_x)
                            # (Lenient matching logic kept) ...
                            if i is None: # Lenient matching Y
                                for formatted_y, idx in y_mapping.items():
                                    if clicked_y in formatted_y or formatted_y in clicked_y: i = idx; break
                            if j is None: # Lenient matching X
                                for formatted_x, idx in x_mapping.items():
                                    if clicked_x in formatted_x or formatted_x in clicked_x: j = idx; break

                            if i is not None and j is not None:
                                reviews = review_matrix[i][j]
                                positive_reviews, negative_reviews, _ = categorize_reviews(reviews)
                                filtered_reviews_for_display = filter_reviews_by_sentiment(reviews, positive_reviews, negative_reviews, sentiment_filter_output)
                                # --- Update Outputs on Success --- #
                                positive_reviews_output = positive_reviews
                                negative_reviews_output = negative_reviews
                                reviews_content_output = create_review_display(
                                    filtered_reviews_for_display, language, plot_type='matrix',
                                    x_category=x_text[j], y_category=y_text[i], selected_words=selected_words
                                )
                                # filter_style_output = get_filter_style() # Set later based on success
                            else:
                                # Simply keep outputs as None/default
                                pass
                        except Exception as e:
                            # Expected in some cases, no need to log anything
                            pass
                    elif process_sentiment_change:
                        # When sentiment filter changes but there's no click_data, this is normal behavior
                        pass
                    else:
                        # Unexpected plot type condition - shouldn't normally happen
                        pass
                else:
                    # Unexpected plot type condition - shouldn't normally happen
                    pass

                # Log action if one was identified (regardless of plot type processing result)
                if log_action:
                    log_user_action(user_real_name, pathname, log_action, log_context)

            except Exception as e:
                # Keep outputs as None/default
                pass
        # --- End Main Processing Block ---

        # --- Final Output Generation --- #
        # Calculate count_display and set filter visibility based on whether processing succeeded
        if positive_reviews_output is not None and negative_reviews_output is not None:
            count_display_output = create_count_display(positive_reviews_output, negative_reviews_output, sentiment_filter_output, language)
            filter_style_output = get_filter_style() # Show filter only if successful
        else:
            # Ensure count is empty and filter is hidden if reviews are None
            count_display_output = ''
            filter_style_output = {'display': 'none'}

        # Return all calculated outputs
        return reviews_content_output, sentiment_filter_output, filter_style_output, count_display_output, selected_words_output

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
         State('date-filter-storage', 'children'),
         State('user-real-name-state', 'data'), # Added state
         State('url', 'pathname')], # Added state
        prevent_initial_call=True
    )
    def update_reviews_by_word_selection(word_buttons_clicks, clear_clicks,
                                        selected_words_json, sentiment_filter, click_data,
                                        language, plot_type, x_value, y_value, top_n_x, top_n_y,
                                        search_query, bar_categories, bar_zoom, bar_count, date_filter_storage,
                                        user_real_name, pathname): # Added states
        """Update the displayed reviews based on word selection"""
        # Check which input triggered the callback
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None # Define trigger_id here

        # If there's no click data, prevent update - we can't filter without a selection
        if not click_data:
            raise dash.exceptions.PreventUpdate

        # Convert selected_words_json to list
        selected_words = json.loads(selected_words_json) if selected_words_json else []

        # Get all available words from the word buttons
        all_words = []
        if ctx.inputs_list and ctx.inputs_list[0]: # Check if inputs_list[0] exists
            for item in ctx.inputs_list[0]:
                if 'id' in item and isinstance(item['id'], dict) and 'index' in item['id']:
                    all_words.append(item['id']['index'])

        # Handle different triggers and prepare logs
        log_action = None
        log_context = {}
        clicked_word = None # Initialize clicked_word

        if trigger_id and 'clear-word-selection-button' in trigger_id:
            log_action = 'clear_word_selection'
            log_context = {'previous_selection': selected_words[:]} # Log previous state
            selected_words = []
        elif trigger_id and 'word-button' in trigger_id and word_buttons_clicks and sum(filter(None, word_buttons_clicks)) > 0:
            try:
                # Try to parse the trigger ID as JSON
                # Ensure trigger_id is a string before splitting
                if isinstance(trigger_id, str):
                    trigger_dict = json.loads(trigger_id.split('.')[0])
                    if trigger_dict.get('type') == 'word-button':
                        clicked_word = trigger_dict.get('index')
                        # Toggle the word selection
                        if clicked_word in selected_words:
                            selected_words.remove(clicked_word)
                            log_action = 'deselect_review_word'
                        else:
                            selected_words.append(clicked_word)
                            log_action = 'select_review_word'

                        log_context = {
                            'word': clicked_word,
                            'current_selection': selected_words[:], # Log current state after change
                            'click_data_source': str(click_data),
                            'sentiment_filter': sentiment_filter,
                            'language': language
                        }
            except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
                pass # Ignore if parsing fails

        # If no user trigger identified, do nothing
        if not log_action:
            raise dash.exceptions.PreventUpdate

        # Parse date range from storage
        date_range = json.loads(date_filter_storage) if date_filter_storage else {"start_date": None, "end_date": None}
        start_date = date_range.get("start_date")
        end_date = date_range.get("end_date")

        # Update reviews with the current word selection
        try:
            reviews_content, _, _, _ = update_reviews_with_filters(
                click_data, sentiment_filter, language, plot_type, x_value, y_value,
                top_n_x, top_n_y, search_query, bar_categories, bar_zoom, bar_count,
                start_date, end_date, selected_words
            )
        except Exception as e:
             print(f"Error in update_reviews_with_filters during word selection: {e}")
             # Fallback or re-raise depending on desired behavior
             raise dash.exceptions.PreventUpdate # Prevent update if review filtering fails

        # Log the action
        log_user_action(user_real_name, pathname, log_action, log_context)

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
        [State('search-help-tooltip', 'style'),
         State('user-real-name-state', 'data'), # Added state
         State('url', 'pathname')], # Added state
        prevent_initial_call=True
    )
    def toggle_search_help(n_clicks, current_style, user_real_name, pathname): # Added states
        if n_clicks:
            action = 'show_search_help' if current_style.get('display') == 'none' else 'hide_search_help'
            log_user_action(user_real_name, pathname, action, {})
        return logic_toggle_search_help(n_clicks, current_style)
    
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
        # Call the logic function from search_logic.py
        return logic_update_search_results_info(
            n_clicks, language, date_slider_value, search_query, 
            plot_type, x_value, y_value, date_filter_storage
        )

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
    def update_date_display_callback(slider_value, date_filter_storage, language, total_reviews):
        return update_date_display(slider_value, date_filter_storage, language, total_reviews)
            
    @app.callback(
        [Output('date-filter-slider', 'value', allow_duplicate=True),
         Output('date-filter-storage', 'children', allow_duplicate=True)],
        [Input('apply-date-btn', 'n_clicks')],
        [State('start-date-input', 'value'),
         State('end-date-input', 'value'),
         State('date-filter-storage', 'children'),
         State('user-real-name-state', 'data'), # Added state
         State('url', 'pathname')], # Added state
        prevent_initial_call=True
    )
    def update_date_from_input_callback(n_clicks, start_date_input, end_date_input, date_filter_storage,
                                         user_real_name, pathname): # Added states
        if n_clicks:
            log_user_action(user_real_name, pathname, 'apply_date_input', {
                'start_date_input': start_date_input,
                'end_date_input': end_date_input
            })
        return update_date_from_input(n_clicks, start_date_input, end_date_input, date_filter_storage)

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
         State('date-filter-storage', 'children'),
         State('user-real-name-state', 'data'), # Add user real name state
         State('url', 'pathname')] # Add pathname state
    )
    def update_trend_chart(plot_type, bar_categories, bar_zoom_value, bar_count, language, n_clicks, 
                          date_slider_value, time_bucket, search_query, date_filter_storage, user_real_name, pathname): # Add states
        """
        Update the trend chart showing category mentions over time.
        """
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
        triggered_by_user = trigger_id and trigger_id not in ['app-language-state.children'] # Add other non-user triggers if needed

        if triggered_by_user:
            log_context = {
                'trigger_id': trigger_id,
                'plot_type': plot_type, # Should always be 'bar_chart' here based on visibility toggle
                'bar_categories': bar_categories,
                'bar_zoom': bar_zoom_value,
                'bar_count': bar_count,
                'time_bucket': time_bucket,
                'search_query': search_query,
                'date_filter_storage': date_filter_storage, # Log the raw storage value
                'language': language
            }
            log_user_action(user_real_name, pathname, 'update_trend_chart', log_context)

        return update_trend_chart_data(plot_type, bar_categories, bar_zoom_value, bar_count, language, 
                                      search_query, date_filter_storage, time_bucket)

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
        # This callback updates options, not a primary user action affecting the view directly. No logging.
        return get_trend_line_options(plot_type, bar_categories, bar_zoom_value, bar_count, language, 
                                     search_query, date_filter_storage, time_bucket)

    @app.callback(
        Output('time-bucket-dropdown', 'options'),
        [Input('app-language-state', 'children')]
    )
    def update_time_bucket_options(language):
        return get_time_bucket_options(language)

    # Add URL pathname callback for login vs main content display
    @app.callback(
        [Output('login-content', 'style', allow_duplicate=True),
         Output('main-content', 'style', allow_duplicate=True)],
        [Input('url', 'pathname')],
        [State('url', 'search'),
         State('app-language-state', 'children'),
         State('category-state', 'children'),
         State('user-real-name-state', 'data')], # Changed children to data
        prevent_initial_call=True
    )
    def display_page(pathname, search, language, category, user_real_name): # Added user_real_name
        """Display login page or main content based on URL pathname and authentication status."""
        # Check if the user is authenticated
        
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
         State('category-state', 'children'),
         State('user-real-name-state', 'data')], # Changed children to data
        prevent_initial_call=True
    )
    def check_auth_and_redirect(pathname, search, language, category, user_real_name): # Added user_real_name
        """Check user authentication and redirect if necessary."""
        
        # If user is not authenticated and tries to access any page other than login, redirect to login
        if pathname != '/login' and not current_user.is_authenticated:
            # Log attempt to access protected route without auth? Maybe too noisy.
            return '/login', f'?lang={language}&category={category}'
            
        # If user is authenticated and tries to access login page, redirect to main page
        if pathname == '/login' and current_user.is_authenticated:
            return '/', f'?lang={language}&category={category}'
        
        # If user is authenticated but real name is missing, force logout and redirect to login
        if pathname != '/login' and current_user.is_authenticated and not user_real_name:
            # Log forced logout due to missing name
            log_user_action(user_real_name, pathname, 'forced_logout_missing_name', {})
            logout_user()
            return '/login', f'?lang={language}&category={category}'
            
        # Otherwise, don't change the URL
        raise dash.exceptions.PreventUpdate