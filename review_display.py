import dash
from dash import html
from collections import Counter
from typing import List, Dict, Optional, Tuple
import numpy as np

from config import TRANSLATIONS, color_mapping
from data import get_bar_chart_data, get_plot_data, raw_dict_map
from utils import format_category_display

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
    ngram_doc_freq = Counter()  # Track document frequency (number of reviews containing ngram)

    # Remove None entries
    filtered_reviews = [r for r in review_dicts if r]

    # Extract text and ngrams from the filtered reviews
    for review_info in filtered_reviews:
        # Extract ngrams based on axis parameter
        if axis == 'x':
            # Only get ngrams from highlight_detail_ngrams for X-axis
            detail_ngrams = [ngram['text'].lower() for ngram in review_info.get('highlight_detail_ngrams', [])]
            all_ngrams.extend(detail_ngrams)
            # Update document frequency
            for ngram in set(detail_ngrams):  # Use set to count each ngram once per review
                ngram_doc_freq[ngram] += 1
        elif axis == 'y':
            # Only get ngrams from highlight_reason_ngrams for Y-axis
            reason_ngrams = [ngram['text'].lower() for ngram in review_info.get('highlight_reason_ngrams', [])]
            all_ngrams.extend(reason_ngrams)
            # Update document frequency
            for ngram in set(reason_ngrams):
                ngram_doc_freq[ngram] += 1
        else:
            # For no specific axis, get all ngrams
            review_ngrams = get_all_ngrams_from_review(review_info)
            all_ngrams.extend(review_ngrams)
            # Update document frequency
            for ngram in set(review_ngrams):
                ngram_doc_freq[ngram] += 1

        # Extract relevant text (highlight or full)
        if axis:
            text = extract_highlight_text(review_info, axis)
        else:
            text = extract_text_from_review(review_info)
        all_words_text.append(text.lower())
    
    # Get IDF values from the loaded data
    ngram_idf = raw_dict_map.get('ngram_idf', {})
    
    # Calculate df*idf scores for each ngram
    ngram_scores = {}
    total_reviews = len(filtered_reviews)
    
    for ngram, df in ngram_doc_freq.items():
        sublinear_df = 1 + np.log(df + 1)
        # Get IDF value, default to 1 if not found
        idf = ngram_idf.get(ngram, 1.0)
        # Calculate final score
        ngram_scores[ngram] = sublinear_df * idf
    
    # Sort ngrams by score and get top n
    sorted_ngrams = sorted(ngram_scores.items(), key=lambda x: x[1], reverse=True)
    top_ngrams = sorted_ngrams[:num_words]
    
    # Return list of (ngram, df) tuples, sorted by df*idf score
    return [(ngram, ngram_doc_freq[ngram]) for ngram, _ in top_ngrams]

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
        all_reviews_words = get_frequent_words(filtered_review_dicts, num_words=50)
        
        # For matrix view, get X and Y axis highlights
        if plot_type == 'matrix':
            x_highlight_words = get_frequent_words(filtered_review_dicts, axis='x', num_words=50)
            y_highlight_words = get_frequent_words(filtered_review_dicts, axis='y', num_words=50)
            
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
            x_highlight_words = get_frequent_words(filtered_review_dicts, axis='x', num_words=50)
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
        
        # Filter all_reviews_words to remove words already in the x or y axis containers
        x_words = set(word for word, count in (x_highlight_words if 'x_highlight_words' in locals() else []))
        y_words = set(word for word, count in (y_highlight_words if 'y_highlight_words' in locals() else []))
        
        # Filter out words that are already in x or y containers
        other_words = [(word, count) for word, count in all_reviews_words if word not in x_words and word not in y_words]
        
        # Add other frequent words not already shown in x/y containers
        if other_words:
            word_freq_components.append(
                create_word_frequency_display(
                    other_words, 
                    language, 
                    'common',
                    on_click_id='other-words-container',
                    selected_words=selected_words,
                    title_word=TRANSLATIONS[language]['other_words'] # Changed title
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
        # Handle matrix view and other matrix-like plot types (use_attr_perf, perf_attr)
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
                    plot_type='matrix',  # Use 'matrix' for the display format
                    x_category=x_text[j],
                    y_category=y_text[i],
                    selected_words=selected_words
                )
                
                return reviews_display, positive_reviews, negative_reviews, None
                
        except ValueError as e:
            print(f"Error processing matrix click data: {str(e)}")
    
    return dash.no_update, None, None, None
