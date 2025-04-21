import dash
from dash import html
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime, timedelta

from config import TRANSLATIONS, AXIS_CATEGORY_NAMES, color_mapping
from data import get_cached_dict, get_plot_data
from utils import ratio_to_rgb, get_size_legend_translations, get_hover_translations, get_proportional_size, format_category_display, get_options


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
    
    # Configure hover info
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
    )
    
    return fig


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
        viz_review_dict_matrix = review_dict_matrix[:, x_sorted_indices]
        viz_x_text = [x_text[i] for i in x_sorted_indices]

        y_sorted_indices = np.argsort(-y_percentages)[:top_n_y]
        viz_matrix = viz_matrix[y_sorted_indices, :]
        viz_sentiment_matrix = viz_sentiment_matrix[y_sorted_indices, :]
        viz_review_dict_matrix = viz_review_dict_matrix[y_sorted_indices, :]
        viz_y_text = [y_text[i] for i in y_sorted_indices]
    else:
        # Handle empty matrix case
        viz_matrix = matrix
        viz_sentiment_matrix = sentiment_matrix
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
        viz_matrix, viz_sentiment_matrix, viz_review_dict_matrix,
        x_axis_text, y_axis_text,
        language, plot_type
    )
    
    return fig, x_options, y_options, max_x, max_y, top_n_x, top_n_y