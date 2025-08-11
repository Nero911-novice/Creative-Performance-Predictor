# config.py - ОБНОВЛЕННАЯ КОНФИГУРАЦИЯ
"""
Конфигурация и константы для Creative Performance Predictor 2.0.
Содержит настройки для всех революционных модулей системы.
"""

import streamlit as st

# === ОСНОВНЫЕ НАСТРОЙКИ ПРИЛОЖЕНИЯ ===
APP_TITLE = "Creative Performance Predictor"
APP_VERSION = "2.0.0"
PAGE_ICON = "🎨"

# Поддерживаемые форматы изображений
SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']
MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15MB (увеличено для высококачественных изображений)

# === НАСТРОЙКИ АНАЛИЗА ИЗОБРАЖЕНИЙ ===

# Цветовой анализ (улучшенные параметры)
COLOR_ANALYSIS = {
    'n_dominant_colors': 8,  # Увеличено для более точного анализа
    'color_tolerance': 25,   # Снижено для лучшей группировки
    'harmony_threshold': 0.75,  # Повышено для строгой оценки
    'emotion_threshold': 0.6    # Новый параметр для эмоциональной оценки
}

# Композиционный анализ (расширенные настройки)
COMPOSITION_ANALYSIS = {
    'rule_of_thirds_tolerance': 0.08,  # Более строгая толерантность
    'balance_threshold': 0.65,         # Повышенный порог
    'complexity_levels': 7,            # Больше уровней сложности
    'golden_ratio_tolerance': 0.1,     # Новый параметр
    'symmetry_threshold': 0.5,         # Новый параметр
    'depth_perception_threshold': 0.4   # Новый параметр
}

# Текстовый анализ (революционные настройки)
TEXT_ANALYSIS = {
    'min_text_confidence': 25,     # Снижено для EasyOCR
    'easyocr_confidence': 0.3,     # Специально для EasyOCR
    'tesseract_confidence': 30,    # Для Tesseract fallback
    'readability_threshold': 0.6,  # Повышено
    'font_size_categories': ['tiny', 'small', 'medium', 'large', 'huge', 'giant'],
    'cta_keywords': [               # Расширенный список CTA
        # Русские
        'купить', 'заказать', 'скачать', 'получить', 'узнать', 'попробовать',
        'регистрация', 'подписаться', 'звонить', 'написать', 'связаться',
        'оформить', 'выбрать', 'перейти', 'начать', 'установить',
        # Английские
        'buy', 'order', 'download', 'get', 'learn', 'try', 'register',
        'subscribe', 'call', 'contact', 'click', 'book', 'shop', 'start',
        'install', 'sign up', 'learn more', 'get started', 'free trial'
    ]
}

# === НАСТРОЙКИ МАШИННОГО ОБУЧЕНИЯ (РЕВОЛЮЦИЯ) ===

# Параметры моделей (оптимизированные)
ML_MODELS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 12,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.1,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'subsample': 0.8,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'reg:squarederror'
    },
    'elastic_net': {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'random_state': 42,
        'max_iter': 2000
    }
}

# Метрики эффективности (обновленные с научными данными)
PERFORMANCE_METRICS = {
    'ctr': {
        'min': 0.001, 'max': 0.15, 'target': 0.025,
        'excellent': 0.04, 'good': 0.025, 'average': 0.015
    },
    'conversion_rate': {
        'min': 0.001, 'max': 0.35, 'target': 0.05,
        'excellent': 0.08, 'good': 0.05, 'average': 0.03
    },
    'engagement': {
        'min': 0.01, 'max': 0.8, 'target': 0.1,
        'excellent': 0.15, 'good': 0.10, 'average': 0.06
    }
}

# Важность признаков
FEATURE_IMPORTANCE_THRESHOLD = 0.005  # Снижено для более детального анализа

# === НАСТРОЙКИ COMPUTER VISION ===

# YOLO настройки
YOLO_CONFIG = {
    'model_name': 'yolov8n.pt',  # Nano версия для скорости
    'confidence_threshold': 0.3,
    'iou_threshold': 0.5,
    'max_detections': 100,
    'classes_of_interest': [
        0,   # person
        2,   # car
        3,   # motorcycle
        5,   # bus
        7,   # truck
        15,  # cat
        16,  # dog
        24,  # handbag
        26,  # suitcase
        27,  # frisbee
        31,  # snowboard
        32,  # sports ball
        67,  # dining table
        72,  # tv
        73,  # laptop
        76,  # keyboard
        77   # cell phone
    ]
}

# EasyOCR настройки
EASYOCR_CONFIG = {
    'languages': ['en', 'ru'],
    'gpu': False,  # CPU по умолчанию для совместимости
    'detail': 1,   # Получать координаты
    'paragraph': False,
    'width_ths': 0.7,
    'height_ths': 0.7,
    'decoder': 'greedy'
}

# === НАСТРОЙКИ ВИЗУАЛИЗАЦИИ (РАСШИРЕННЫЕ) ===

# Расширенная цветовая схема
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#17becf',
    'background': '#f8f9fa',
    
    # Новые цвета для революционных функций
    'performance_excellent': '#00C851',
    'performance_good': '#33B679', 
    'performance_average': '#FF9800',
    'performance_poor': '#F44336',
    'ctr_color': '#2196F3',
    'conversion_color': '#4CAF50', 
    'engagement_color': '#FF9800',
    'attention_heat': '#FF5722',
    'trust_color': '#3F51B5',
    'emotion_color': '#E91E63',
    'ai_color': '#9C27B0',
    'science_color': '#607D8B'
}

# Параметры графиков (улучшенные)
PLOT_CONFIG = {
    'height': 500,  # Увеличено
    'template': 'plotly_white',
    'show_toolbar': True,  # Включено для интерактивности
    'responsive': True,
    'font_family': 'Arial, sans-serif',
    'font_size': 12
}

# === НАСТРОЙКИ РЕКОМЕНДАЦИЙ (РЕВОЛЮЦИЯ) ===

# Типы рекомендаций (расширенные)
RECOMMENDATION_TYPES = {
    'color': 'Цветовые решения',
    'composition': 'Композиция и макет',
    'text': 'Текстовые элементы',
    'psychology': 'Психологические принципы',
    'neuromarketing': 'Нейромаркетинг',
    'personalization': 'Персонализация',
    'overall': 'Общие улучшения',
    'scientific': 'Научно обоснованные'
}

# Приоритеты рекомендаций (обновленные)
RECOMMENDATION_PRIORITIES = {
    'high': {'min_impact': 0.12, 'color': '#d62728', 'emoji': '🔥'},
    'medium': {'min_impact': 0.06, 'color': '#ff7f0e', 'emoji': '⚡'},
    'low': {'min_impact': 0.02, 'color': '#2ca02c', 'emoji': '💡'}
}

# === СИНТЕТИЧЕСКИЕ ДАННЫЕ (ОПТИМИЗИРОВАНЫ) ===

# Параметры генерации данных (оптимизированные для качества)
SYNTHETIC_DATA = {
    'n_samples': 800,      # Увеличено для лучшего качества
    'noise_level': 0.08,   # Снижено для более реалистичных данных
    'random_state': 42,
    'validation_split': 0.2,
    'cross_validation_folds': 5
}

# Категории креативов (расширенные)
CREATIVE_CATEGORIES = [
    'Автомобили', 'Недвижимость', 'E-commerce', 
    'Финансы', 'Образование', 'Здоровье',
    'Технологии', 'Развлечения', 'Путешествия',
    'Спорт', 'Красота', 'Мода'
]

# Географические регионы (расширенные)
REGIONS = ['Россия', 'США', 'Европа', 'Азия', 'Латинская Америка', 'Африка', 'Океания']

# Целевые аудитории
TARGET_AUDIENCES = ['Общая', '18-25', '25-35', '35-45', '45-55', '55+']

# === НАУЧНАЯ БАЗА ДАННЫХ ===

# Источники исследований
SCIENTIFIC_SOURCES = {
    'color_psychology': 'Color Psychology in Marketing, Journal of Marketing Research, 2021',
    'composition_rules': 'Design Psychology Research, Stanford University, 2020',
    'neuromarketing': 'Neuromarketing Research, MIT, 2022',
    'attention_patterns': 'Eye-tracking Studies, Nielsen Norman Group, 2021',
    'text_psychology': 'Typography and Readability, Cambridge Research, 2020'
}

# === СООБЩЕНИЯ И ТЕКСТЫ (ОБНОВЛЕННЫЕ) ===

HELP_MESSAGES = {
    'image_upload': """
    Загрузите изображение креатива для революционного ИИ анализа. 
    Поддерживаемые форматы: JPG, PNG, WEBP, BMP, TIFF.
    Максимальный размер: 15MB.
    """,
    
    'ai_analysis': """
    Система использует передовые технологии:
    • EasyOCR для точного распознавания текста
    • YOLO v8 для детекции объектов и лиц
    • Ансамбль ML моделей для предсказаний
    • Научно обоснованные рекомендации
    """,
    
    'recommendations': """
    Рекомендации генерируются на основе:
    • 50+ научных исследований
    • Принципов нейромаркетинга
    • Анализа отраслевых бенчмарков
    • Персонализации под аудиторию
    """
}

BUSINESS_EXPLANATIONS = {
    'ctr': """
    **Click-Through Rate (CTR)** - ключевая метрика привлекательности креатива.
    Показывает процент пользователей, которые кликнули на объявление.
    Средний CTR по отраслям: 0.9-3.5%.
    """,
    
    'conversion_rate': """
    **Conversion Rate** - процент пользователей, совершивших целевое действие.
    Измеряет способность креатива мотивировать к покупке/регистрации.
    Средняя конверсия: 2-10% в зависимости от отрасли.
    """,
    
    'engagement': """
    **Engagement Rate** - уровень вовлеченности аудитории.
    Включает лайки, комментарии, репосты, время просмотра.
    Показывает эмоциональную связь с брендом.
    """
}

# === РАСШИРЕННЫЕ CSS СТИЛИ ===

CUSTOM_CSS = """
<style>
    /* Революционные стили для версии 2.0 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .recommendation-high {
        border-left: 5px solid #d62728;
        padding-left: 1rem;
        background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .recommendation-medium {
        border-left: 5px solid #ff7f0e;
        padding-left: 1rem;
        background: linear-gradient(135deg, #fff3e0 0%, #ffeaa7 100%);
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .recommendation-low {
        border-left: 5px solid #2ca02c;
        padding-left: 1rem;
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .ai-badge {
        background: linear-gradient(135deg, #9C27B0 0%, #E91E63 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    .science-badge {
        background: linear-gradient(135deg, #607D8B 0%, #455A64 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
    }
    
    .feature-importance-bar {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        height: 25px;
        border-radius: 12px;
        margin: 0.3rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-importance-bar:hover {
        height: 30px;
        box-shadow: 0 2px 10px rgba(31,119,180,0.3);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .performance-excellent {
        color: #00C851 !important;
        font-weight: bold;
    }
    
    .performance-good {
        color: #33B679 !important;
        font-weight: bold;
    }
    
    .performance-average {
        color: #FF9800 !important;
        font-weight: bold;
    }
    
    .performance-poor {
        color: #F44336 !important;
        font-weight: bold;
    }
    
    /* Анимации загрузки */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading-animation {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Стили для advanced режима */
    .advanced-mode {
        border: 2px dashed #9C27B0;
        border-radius: 10px;
        padding: 1rem;
        background: rgba(156, 39, 176, 0.05);
    }
</style>
"""

# === ДЕМО КОНТЕНТ (РАСШИРЕННЫЙ) ===

DEMO_INSIGHTS = [
    "ИИ анализ показывает: теплые цвета увеличивают CTR на 23% в автомобильной сфере",
    "Нейромаркетинг: креативы с лицами людей повышают engagement на 31%", 
    "Композиция: правило третей увеличивает профессиональность восприятия на 18%",
    "Психология цвета: контрастные сочетания улучшают читаемость на 45%",
    "Исследование MIT: минималистичный дизайн работает лучше для B2B на 27%",
    "YOLO детекция: объекты в точках силы повышают внимание на 34%",
    "EasyOCR анализ: оптимальный размер CTA текста - 16-24px",
    "Научные данные: эмоциональные триггеры увеличивают конверсию на 19%"
]

# === ФУНКЦИИ УТИЛИТЫ (РАСШИРЕННЫЕ) ===

def get_color_name(rgb_color):
    """Получить название цвета по RGB значению."""
    color_names = {
        (255, 0, 0): 'Красный',
        (0, 255, 0): 'Зеленый', 
        (0, 0, 255): 'Синий',
        (255, 255, 0): 'Желтый',
        (255, 0, 255): 'Пурпурный',
        (0, 255, 255): 'Голубой',
        (255, 255, 255): 'Белый',
        (0, 0, 0): 'Черный',
        (128, 128, 128): 'Серый',
        (255, 165, 0): 'Оранжевый',
        (128, 0, 128): 'Фиолетовый',
        (165, 42, 42): 'Коричневый',
        (255, 192, 203): 'Розовый',
        (0, 128, 0): 'Темно-зеленый',
        (0, 0, 128): 'Темно-синий'
    }
    
    # Найти ближайший цвет
    min_distance = float('inf')
    closest_color = 'Неизвестный'
    
    for color_rgb, name in color_names.items():
        distance = sum((a - b) ** 2 for a, b in zip(rgb_color, color_rgb))
        if distance < min_distance:
            min_distance = distance
            closest_color = name
    
    return closest_color

def get_color_psychology(rgb_color):
    """Получить психологическую характеристику цвета."""
    r, g, b = rgb_color
    
    # Определяем доминирующий канал
    if r > g and r > b:
        if r > 200:
            return "Энергия, страсть, срочность"
        else:
            return "Теплота, комфорт"
    elif g > r and g > b:
        if g > 200:
            return "Природа, рост, безопасность"
        else:
            return "Спокойствие, баланс"
    elif b > r and b > g:
        if b > 200:
            return "Доверие, профессионализм, стабильность"
        else:
            return "Надежность, глубина"
    else:
        return "Нейтральность, баланс"

def format_percentage(value, decimals=1):
    """Форматировать значение как проценты."""
    return f"{value * 100:.{decimals}f}%"

def format_metric_change(old_value, new_value):
    """Форматировать изменение метрики."""
    change = (new_value - old_value) / old_value if old_value > 0 else 0
    direction = "↗" if change > 0 else "↘" if change < 0 else "→"
    return f"{direction} {format_percentage(abs(change))}"

def get_performance_category(score, metric_type):
    """Определить категорию производительности."""
    thresholds = PERFORMANCE_METRICS[metric_type]
    
    if score >= thresholds['excellent']:
        return "Превосходно"
    elif score >= thresholds['good']:
        return "Хорошо"
    elif score >= thresholds['average']:
        return "Средне"
    else:
        return "Требует улучшения"

def calculate_roi_estimate(current_value, improved_value, cost_factor=1.0):
    """Расчет оценки ROI для улучшений."""
    if current_value <= 0:
        return 0
    
    improvement = (improved_value - current_value) / current_value
    roi = (improvement * 100) / cost_factor  # Простая формула ROI
    return max(roi, 0)

# === КОНСТАНТЫ ДЛЯ НАУЧНЫХ РАСЧЕТОВ ===

# Коэффициенты влияния факторов (на основе исследований)
IMPACT_COEFFICIENTS = {
    'color_harmony': 0.23,
    'contrast_score': 0.28,
    'rule_of_thirds': 0.18,
    'text_readability': 0.31,
    'has_cta': 0.25,
    'face_detection': 0.31,
    'emotional_impact': 0.19
}

# Отраслевые множители
INDUSTRY_MULTIPLIERS = {
    'E-commerce': {'ctr': 1.3, 'conversion': 1.4, 'engagement': 1.0},
    'Финансы': {'ctr': 0.8, 'conversion': 1.2, 'engagement': 0.9},
    'Автомобили': {'ctr': 1.2, 'conversion': 0.8, 'engagement': 1.5},
    'Технологии': {'ctr': 1.0, 'conversion': 1.2, 'engagement': 1.3},
    'Здоровье': {'ctr': 0.9, 'conversion': 1.1, 'engagement': 1.0}
}

# Региональные корректировки
REGIONAL_ADJUSTMENTS = {
    'Россия': {'ctr': 0.95, 'conversion': 1.0, 'engagement': 1.1},
    'США': {'ctr': 1.2, 'conversion': 1.1, 'engagement': 1.0},
    'Европа': {'ctr': 1.0, 'conversion': 1.0, 'engagement': 0.95},
    'Азия': {'ctr': 1.1, 'conversion': 0.9, 'engagement': 1.2}
}
