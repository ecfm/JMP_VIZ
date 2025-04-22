import os
from cachetools import TTLCache

# Constants and directory setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_result_dir(category='Cables'):
    """Get the result directory based on the category."""
    return os.path.join(ROOT_DIR, f"result/{category}")

# Authentication configuration
VALID_USERNAME = "xsight"
VALID_PASSWORD = "mit100"

# Cache configurations
CACHE_TTL = 3600  # Cache timeout in seconds (1 hour)
CACHE_MAXSIZE = 100  # Maximum number of items in cache

# Color mappings for highlighting
color_mapping = {
    '+': '#E6F3FF',  # Blue background for positive
    '-': '#FFE6E6',  # Red background for negative
    '?': '#FFFFFF',  # White background for neutral
    'x': '#805AD5',  # Purple border for X-axis
    'y': '#38A169',  # Green border for Y-axis
}

# Define category type colors for mapping
type_colors = {
    'U: ': '#6A0DAD',  # Dark Purple for Usage (changed from Blue #4285F4)
    'A: ': '#34A853',  # Green for Attribute
    'P: ': '#FBBC05',  # Yellow for Performance
}

color_to_prefix = {v: k for k, v in type_colors.items()}

# Highlight examples
def get_highlight_examples(language='en'):
    if language == 'en':
        pos_highlight = f'<span style="background-color: {color_mapping["+"]}; color: black; padding: 5px;">Blue background indicates positive sentiment</span>'
        neg_highlight = f'<span style="background-color: {color_mapping["-"]}; color: black; padding: 5px;">Red background indicates negative sentiment</span>'
    else:  # zh
        pos_highlight = f'<span style="background-color: {color_mapping["+"]}; color: black; padding: 5px;">蓝底表示满意</span>'
        neg_highlight = f'<span style="background-color: {color_mapping["-"]}; color: black; padding: 5px;">红底表示不满意</span>'
    
    return pos_highlight, neg_highlight

def get_review_highlight(language='en', plot_type='matrix', x_category=None, y_category=None):
    """
    Get the appropriate review highlight based on language and plot type
    """
    x_highlight = ''
    y_highlight = ''
    if x_category:
        if language == 'en':
            x_highlight = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["x"]}; color: black; padding: 5px;">Text related to {x_category}</span>'
        else:  # zh
            x_highlight = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["x"]}; color: black; padding: 5px;">{x_category}相关文本</span>'
    
    if y_category:
        if language == 'en':
            y_highlight = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["y"]}; color: black; padding: 5px;">Text related to {y_category}</span>'
        else:  # zh
            y_highlight = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["y"]}; color: black; padding: 5px;">{y_category}相关文本</span>'
        
    return x_highlight, y_highlight

def get_review_format(language='en', plot_type='matrix', x_category=None, y_category=None):
    """
    Get the appropriate review format based on language and plot type
    
    Args:
        language: 'en' or 'zh'
        plot_type: 'bar_chart' or 'matrix'
        x_category: The selected X-axis category (optional)
        y_category: The selected Y-axis category (optional)
        
    Returns:
        The formatted review format string
    """ 
    x_highlight, y_highlight = get_review_highlight(language, plot_type, x_category, y_category)
    pos_highlight, neg_highlight = get_highlight_examples(language)
    if plot_type == 'bar_chart':
        # For bar chart, exclude the Y-axis highlight
        if language == 'en':
            return f'Display Format: Review: {x_highlight} {pos_highlight} {neg_highlight} ASIN (Star Rating) <b>Review Title</b> Review Text'
        else:  # zh
            return f'显示格式： 评论: {x_highlight} {pos_highlight} {neg_highlight} ASIN (评论星级) <b>评论标题</b> 评论内容'
    else:  # matrix
        # For matrix view, include both X and Y highlights
        if language == 'en':
            return f'Display Format: Review: {x_highlight} {y_highlight} {pos_highlight} {neg_highlight} ASIN (Star Rating) <b>Review Title</b> Review Text'
        else:  # zh
            return f'显示格式： 评论: {x_highlight} {y_highlight} {pos_highlight} {neg_highlight} ASIN (评论星级) <b>评论标题</b> 评论内容'

# Language translations
TRANSLATIONS = {
    'en': {
        'login': 'Login',
        'username': 'Username',
        'password': 'Password',
        'real_name': 'Name',
        'enter_username': 'Enter username',
        'enter_password': 'Enter password',
        'enter_real_name': 'Enter your name',
        'invalid_credentials': 'Invalid credentials',
        'logout': 'Logout',
        'plot_type': 'Plot Type:',
        'use_vs_attr_perf': 'Use v.s. Attribute + Performance',
        'perf_vs_attr': 'Performance v.s. Attribute',
        'y_axis_category': 'Y-axis Category:',
        'x_axis_category': 'X-axis Category:',
        'num_y_features': 'Number of Y Features:',
        'num_x_features': 'Number of X Features:',
        'all_level_0': 'All [Level 0]',
        'reviews': 'Reviews',
        'no_reviews': 'No reviews available for this selection.',
        'selected': 'Selected:',
        'review': 'Review',
        'satisfaction_level': 'Satisfaction Level',
        'unsatisfied': 'Unsatisfied',
        'neutral': 'Neutral',
        'satisfied': 'Satisfied',
        'use_attr_perf_title': 'Usage v.s. Product Performance and Attributes',
        'perf_attr_title': 'Performance v.s. Attributes',
        'percentage_explanation': 'Percentage of total mentions',
        'x_axis_percentage': 'Percentage of total mentions',
        'y_axis_percentage': 'Percentage of total mentions',
        'search_placeholder': 'e.g. "memory card" & (good great)',
        'sentiment_filter': 'Filter Reviews:',
        'show_all': 'Show All',
        'show_positive': 'Show Positive',
        'show_negative': 'Show Negative',
        'positive_count': 'Positive Reviews: {}',
        'negative_count': 'Negative Reviews: {}',
        'size_explanation': "Square size is proportional to number of mentions",
        'hover_x': 'X-axis',
        'hover_y': 'Y-axis',
        'hover_count': 'Count',
        'hover_satisfaction': 'Satisfaction',
        'hover_date': 'Date',
        'period': 'Time period',
        'search_results': "Found {} reviews matching query: {}",
        'review_format': 'Display Format: Review x: <PLACEHOLDER_X_HIGHLIGHT> <PLACEHOLDER_Y_HIGHLIGHT> <PLACEHOLDER_POS_HIGHLIGHT> <PLACEHOLDER_NEG_HIGHLIGHT> ASIN (Star Rating) <b>Review Title</b> Review Text',
        'bar_chart': 'Common Topics in Reviews',
        'usage_category': 'Usage',
        'attribute_category': 'Attribute',
        'performance_category': 'Performance',
        'select_category': 'Select a category to zoom in',
        'bar_category_label': 'Categories to include:',
        'bar_zoom_label': 'Zoom in on category:',
        'bar_chart_title': 'Frequency of Common Topics in Reviews',
        'category_types': 'Category Types',
        'bar_count_label': 'Number of categories to show:',
        'select_all': 'Select All',
        'unselect_all': 'Unselect All',
        'date_filter_label': 'Filter by review date:',
        'date_range_format': '%Y-%m-%d',
        'total_reviews': 'Total Reviews',
        'total_reviews_filtered': 'Total Reviews Matching Filter',
        'selected_date_range': 'Selected date range',
        'search': 'Search',
        'months': 'months',
        'trend_chart_title': 'Frequency of Common Topics in Reviews Over Time',
        'trend_x_axis': 'Date',
        'trend_y_axis': 'Mention Count',
        'hover_date': 'Date',
        'no_data_available': 'No data available',
        'time_bucket_label': 'Time interval:',
        'trend_line_selector_label': 'Select trend lines to display:',
        'month': 'Month',
        'three_month': '3 Months',
        'six_month': '6 Months',
        'year': 'Year',
        'frequent_words': 'Frequent Words',
        'frequent_positive_words': 'Frequent Positive Words',
        'frequent_negative_words': 'Frequent Negative Words',
        'select_all_words': 'Select All Words',
        'deselect_all_words': 'Deselect All Words',
        'showing_reviews_with': 'Showing reviews containing:',
        'showing_all_reviews': 'Showing all reviews',
        'no_word_selected': 'No word selected',
        'clear_word_selection': 'Clear selection',
        'all_reviews': 'All Reviews',
    },
    'zh': {
        'login': '登录',
        'username': '账户',
        'password': '密码',
        'real_name': '姓名',
        'enter_username': '请输入用户名',
        'enter_password': '请输入密码',
        'enter_real_name': '输入您的姓名',
        'invalid_credentials': '用户名或密码错误',
        'logout': '退出登录',
        'plot_type': '图表类型：',
        'use_vs_attr_perf': '用途 vs. 属性 + 性能',
        'perf_vs_attr': '性能 vs. 属性',
        'y_axis_category': 'Y轴类别：',
        'x_axis_category': 'X轴类别：',
        'num_y_features': 'Y类别数量：',
        'num_x_features': 'X类别数量：',
        'all_level_0': '全部 [L0]',
        'reviews': '评论',
        'no_reviews': '该选择没有可用的评论。',
        'selected': '已选择：',
        'review': '评论',
        'satisfaction_level': '满意度',
        'unsatisfied': '不满意',
        'neutral': '中性',
        'satisfied': '满意',
        'use_attr_perf_title': '用途 vs. 产品属性和性能',
        'perf_attr_title': '性能 vs. 属性',
        'percentage_explanation': '占总提及次数的百分比',
        'x_axis_percentage': '(该类别被提及次数%)',
        'y_axis_percentage': '(该类别被提及次数%)',
        'search_placeholder': '例如："memory card" & (good great)',
        'sentiment_filter': '评论筛选：',
        'show_all': '显示全部',
        'show_positive': '显示正面评论',
        'show_negative': '显示负面评论',
        'positive_count': '正面评论数: {}',
        'negative_count': '负面评论数: {}',
        'size_explanation': "方块大小与提及次数成正比",
        'hover_x': 'X轴',
        'hover_y': 'Y轴',
        'hover_count': '数量',
        'hover_satisfaction': '满意度',
        'hover_date': '日期',
        'period': '时间段',
        'search_results': "找到 {} 条匹配的评论，搜索条件：{}",
        'review_format': f'显示格式： 评论 x: <PLACEHOLDER_X_HIGHLIGHT> <PLACEHOLDER_Y_HIGHLIGHT> <PLACEHOLDER_POS_HIGHLIGHT> <PLACEHOLDER_NEG_HIGHLIGHT> ASIN (评论星级) <b>评论标题</b> 评论内容',
        'bar_chart': '各类信息提及频率',
        'usage_category': '用途',
        'attribute_category': '属性',
        'performance_category': '性能',
        'select_category': '选择要显示的类别',
        'bar_category_label': '包含的类别：',
        'bar_zoom_label': '显示的类别：',
        'bar_chart_title': '各类信息提及频率',
        'category_types': '类别类型',
        'bar_count_label': '显示的类别数量:',
        'select_all': '全选',
        'unselect_all': '全不选',
        'date_filter_label': '按评论日期过滤:',
        'date_range_format': '%Y-%m-%d',
        'total_reviews': '总评论数',
        'total_reviews_filtered': '符合筛选条件的评论数',
        'selected_date_range': '选定日期范围',
        'search': '搜索',
        'months': '个月',
        'trend_chart_title': '各类信息提及频率随时间变化',
        'trend_x_axis': '日期',
        'trend_y_axis': '提及次数',
        'hover_date': '日期',
        'no_data_available': '没有可用数据',
        'time_bucket_label': '时间间隔:',
        'trend_line_selector_label': '选择要显示的趋势线:',
        'month': '月',
        'three_month': '3个月',
        'six_month': '6个月',
        'year': '年',
        'frequent_words': '高频词',
        'frequent_positive_words': '高频正面词',
        'frequent_negative_words': '高频负面词',
        'select_all_words': '全选词',
        'deselect_all_words': '全不选词',
        'showing_reviews_with': '显示包含:',
        'showing_all_reviews': '显示所有评论',
        'no_word_selected': '没有选词',
        'clear_word_selection': '清除选择',
        'all_reviews': '所有评论',
    }
}

# Instead of static values, we'll make these callable functions
TRANSLATIONS['en']['get_bar_chart_review_format'] = lambda x_cat=None: get_review_format('en', 'bar_chart', x_cat)
TRANSLATIONS['en']['get_matrix_review_format'] = lambda x_cat=None, y_cat=None: get_review_format('en', 'matrix', x_cat, y_cat)
TRANSLATIONS['zh']['get_bar_chart_review_format'] = lambda x_cat=None: get_review_format('zh', 'bar_chart', x_cat)
TRANSLATIONS['zh']['get_matrix_review_format'] = lambda x_cat=None, y_cat=None: get_review_format('zh', 'matrix', x_cat, y_cat)

# Axis category names translations
AXIS_CATEGORY_NAMES = {
    'en': {
        'use': 'Usage',
        'perf': 'Performance',
        'attr': 'Attributes',
        'attr_perf': 'Attributes & Performance'
    },
    'zh': {
        'use': '用途',
        'perf': '性能',
        'attr': '属性',
        'attr_perf': '属性与性能'
    }
}

# Search examples translations
SEARCH_EXAMPLES = {
    'en': [
        "Search syntax: Use & (AND), space (OR), () for grouping. Use quotes for multi-word terms. Case insensitive.",
        "Examples:",
        "- B08NJ2SGDW B09NTR4KKL: search for reviews with asin B08NJ2SGDW or B09NTR4KKL",
        "- 'memory card' & B09NTR4KKL: search for reviews with 'memory card' and asin B09NTR4KKL"
    ],
    'zh': [
        "搜索语法：使用 & (且)，空格 (或)，() 用于分组。多词短语请加引号。不区分大小写。",
        "示例：",
        "- B08NJ2SGDW B09NTR4KKL: 搜索asin为 B08NJ2SGDW 或 B09NTR4KKL 的评论",
        '- "memory card" & B09NTR4KKL: 搜索包含"内存卡"这个词组并且 asin 为 B09NTR4KKL 的评论'
    ]
}

# Create cache instances
path_dict_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)
plot_data_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL) 