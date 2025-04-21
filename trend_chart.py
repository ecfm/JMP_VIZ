"""
Trend chart functionality for the visualization app.
Contains functions for creating and updating trend charts.
"""

import json
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from data import get_bar_chart_data, get_category_time_series
from utils import ratio_to_rgb, format_category_display
from config import type_colors, TRANSLATIONS


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


def update_trend_chart_data(plot_type, bar_categories, bar_zoom_value, bar_count, language, search_query, date_filter_storage, time_bucket):
    """Process data for trend chart and prepare it for visualization"""
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


def get_trend_line_options(plot_type, bar_categories, bar_zoom_value, bar_count, language, search_query, date_filter_storage, time_bucket):
    """Get options for the trend line selector"""
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
            for p in type_colors.keys():
                if category.startswith(p):
                    break
            
            # Format the label similar to other parts of the app
            formatted_category = format_category_display(category, language)
            
            valid_options.append({
                'label': formatted_category,
                'value': category
            })
            initial_values.append(category)
    
    return valid_options, initial_values


def get_time_bucket_options(language):
    """Get time bucket dropdown options"""
    return [
        {'label': TRANSLATIONS[language].get('month', 'Month'), 'value': 'month'},
        {'label': TRANSLATIONS[language].get('three_month', '3 Months'), 'value': '3month'},
        {'label': TRANSLATIONS[language].get('six_month', '6 Months'), 'value': '6month'},
        {'label': TRANSLATIONS[language].get('year', 'Year'), 'value': 'year'}
    ] 