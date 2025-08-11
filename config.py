# config.py
"""
Конфигурация и константы для Creative Performance Predictor.
Содержит настройки для всех модулей системы.
"""

import streamlit as st

# === ОСНОВНЫЕ НАСТРОЙКИ ПРИЛОЖЕНИЯ ===
APP_TITLE = "Creative Performance Predictor"
APP_VERSION = "1.0.0"
PAGE_ICON = "🎨"

# Поддерживаемые форматы изображений
SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'webp', 'bmp']
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

# === НАСТРОЙКИ АНАЛИЗА ИЗОБРАЖЕНИЙ ===

# Цветовой анализ
COLOR_ANALYSIS = {
    'n_dominant_colors': 5,  # Количество доминирующих цветов
    'color_tolerance': 30,   # Толерантность для группировки цветов
    'harmony_threshold': 0.7 # Порог для цветовой гармонии
}

# Композиционный анализ
COMPOSITION_ANALYSIS = {
    'rule_of_thirds_tolerance': 0.1,  # Толерантность для правила третей
    'balance_threshold': 0.6,         # Порог визуального баланса
    'complexity_levels': 5            # Уровни сложности композиции
}

# Текстовый анализ
TEXT_ANALYSIS = {
    'min_text_confidence': 30,   # Минимальная уверенность OCR
    'readability_threshold': 0.5, # Порог читаемости
    'font_size_categories': ['small', 'medium', 'large', 'huge']
}

# === НАСТРОЙКИ МАШИННОГО ОБУЧЕНИЯ ===

# Параметры моделей
ML_MODELS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'linear': {
        'alpha': 1.0,
        'random_state': 42
    }
}

# Метрики эффективности
PERFORMANCE_METRICS = {
    'ctr': {'min': 0.001, 'max': 0.1, 'target': 0.02},
    'conversion_rate': {'min': 0.001, 'max': 0.5, 'target': 0.05},
    'engagement': {'min': 0.01, 'max': 1.0, 'target': 0.1}
}

# Важность признаков
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# === НАСТРОЙКИ ВИЗУАЛИЗАЦИИ ===

# Цветовая схема приложения
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#17becf',
    'background': '#f8f9fa'
}

# Параметры графиков
PLOT_CONFIG = {
    'height': 400,
    'template': 'plotly_white',
    'show_toolbar': False,
    'responsive': True
}

# === НАСТРОЙКИ РЕКОМЕНДАЦИЙ ===

# Типы рекомендаций
RECOMMENDATION_TYPES = {
    'color': 'Цветовые решения',
    'composition': 'Композиция',
    'text': 'Текстовые элементы',
    'overall': 'Общие улучшения'
}

# Приоритеты рекомендаций
RECOMMENDATION_PRIORITIES = {
    'high': {'min_impact': 0.15, 'color': '#d62728'},
    'medium': {'min_impact': 0.08, 'color': '#ff7f0e'},
    'low': {'min_impact': 0.03, 'color': '#2ca02c'}
}

# === СИНТЕТИЧЕСКИЕ ДАННЫЕ ===

# Параметры генерации данных (оптимизированные для скорости)
SYNTHETIC_DATA = {
    'n_samples': 300,      # Уменьшено с 1000 для быстрого обучения
    'noise_level': 0.1,
    'random_state': 42
}

# Категории креативов
CREATIVE_CATEGORIES = [
    'Автомобили', 'Недвижимость', 'E-commerce', 
    'Финансы', 'Образование', 'Здоровье',
    'Технологии', 'Развлечения', 'Путешествия'
]

# Географические регионы
REGIONS = ['Россия', 'США', 'Европа', 'Азия', 'Другие']

# === СООБЩЕНИЯ И ТЕКСТЫ ===

HELP_MESSAGES = {
    'image_upload': """
    Загрузите изображение креатива для анализа. 
    Поддерживаемые форматы: JPG, PNG, WEBP.
    Максимальный размер: 10MB.
    """,
    
    'model_prediction': """
    Модель анализирует визуальные характеристики креатива
    и предсказывает его эффективность на основе обученных паттернов.
    """,
    
    'recommendations': """
    Рекомендации генерируются на основе анализа важности признаков
    и статистических паттернов успешных креативов.
    """
}

BUSINESS_EXPLANATIONS = {
    'ctr': """
    **Click-Through Rate (CTR)** - отношение кликов к показам.
    Показывает, насколько креатив привлекает внимание аудитории.
    """,
    
    'conversion_rate': """
    **Conversion Rate** - отношение конверсий к кликам.
    Измеряет способность креатива мотивировать к действию.
    """,
    
    'engagement': """
    **Engagement Rate** - уровень вовлеченности аудитории.
    Включает лайки, комментарии, репосты и время просмотра.
    """
}

# === CSS СТИЛИ ===

CUSTOM_CSS = """
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .recommendation-high {
        border-left: 4px solid #d62728;
        padding-left: 1rem;
        background-color: #ffebee;
    }
    
    .recommendation-medium {
        border-left: 4px solid #ff7f0e;
        padding-left: 1rem;
        background-color: #fff3e0;
    }
    
    .recommendation-low {
        border-left: 4px solid #2ca02c;
        padding-left: 1rem;
        background-color: #e8f5e8;
    }
    
    .feature-importance-bar {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        height: 20px;
        border-radius: 10px;
        margin: 0.2rem 0;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
"""

# === ДЕМО КОНТЕНТ ===

DEMO_INSIGHTS = [
    "Теплые цвета показывают на 23% выше CTR в сфере автомобилей",
    "Креативы с лицами людей увеличивают engagement на 31%", 
    "Правило третей повышает восприятие профессиональности на 18%",
    "Контрастные цвета улучшают читаемость текста на 45%",
    "Минималистичная композиция работает лучше для B2B сегмента"
]

SAMPLE_RECOMMENDATIONS = {
    'color': [
        "Рассмотрите использование более теплых оттенков для увеличения эмоциональной привлекательности",
        "Повысьте контрастность между текстом и фоном для лучшей читаемости",
        "Добавьте акцентный цвет для привлечения внимания к CTA"
    ],
    'composition': [
        "Переместите основной объект в точку пересечения линий правила третей",
        "Уменьшите визуальную сложность для лучшего восприятия",
        "Добавьте больше свободного пространства вокруг ключевых элементов"
    ],
    'text': [
        "Увеличьте размер основного заголовка для лучшей иерархии",
        "Сократите количество текста для мобильных устройств",
        "Используйте более контрастные цвета для текстовых элементов"
    ]
}

# === ФУНКЦИИ УТИЛИТЫ ===

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
        (128, 128, 128): 'Серый'
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

def format_percentage(value, decimals=1):
    """Форматировать значение как проценты."""
    return f"{value * 100:.{decimals}f}%"

def format_metric_change(old_value, new_value):
    """Форматировать изменение метрики."""
    change = (new_value - old_value) / old_value
    direction = "↗" if change > 0 else "↘" if change < 0 else "→"
    return f"{direction} {format_percentage(abs(change))}"
