# visualizer.py
"""
Модуль визуализации для Creative Performance Predictor.
Создает интерактивные графики и диаграммы для анализа креативов.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image, ImageDraw, ImageFont

from config import COLOR_SCHEME, PLOT_CONFIG, get_color_name

class Visualizer:
    """
    Класс для создания визуализаций анализа креативов.
    Включает графики анализа, предсказаний и рекомендаций.
    """
    
    def __init__(self):
        self.color_scheme = COLOR_SCHEME
        self.plot_config = PLOT_CONFIG
        
    def plot_color_analysis(self, color_data: Dict[str, Any]) -> go.Figure:
        """Создание графика анализа цветовых характеристик."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Доминирующие цвета', 'Ключевые цветовые метрики', 'Цветовая палитра', 'Цветовой профиль'],
            specs=[[{'type': 'bar'}, {'type': 'indicator'}],
                   [{'type': 'scatter', 'colspan': 2}, None]],
            vertical_spacing=0.3
        )
        
        # 1. Доминирующие цвета
        if 'dominant_colors' in color_data and color_data['dominant_colors']:
            colors = color_data['dominant_colors'][:5]
            color_names = [get_color_name(c) for c in colors]
            color_hex = [f'rgb({c[0]},{c[1]},{c[2]})' for c in colors]
            
            fig.add_trace(go.Bar(
                x=color_names, y=[1] * len(colors), marker_color=color_hex,
                name='Доминирующие цвета', showlegend=False
            ), row=1, col=1)

        # 2. Индикатор общей оценки цвета
        # УЛУЧШЕНИЕ: Упрощено до одного более читаемого индикатора
        avg_score = (color_data.get('brightness', 0) + color_data.get('saturation', 0) + 
                     color_data.get('contrast_score', 0) + color_data.get('harmony_score', 0)) / 4
        
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=avg_score * 100,
            title={'text': "Качество цвета"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': self.color_scheme['primary']}},
            number={'suffix': "%"}
        ), row=1, col=2)

        # 3. Цветовая палитра (было scatter, стало более наглядное отображение)
        # УЛУЧШЕНИЕ: Вместо непонятной точки, показываем сами цвета
        if 'dominant_colors' in color_data and color_data['dominant_colors']:
            palette_colors = color_data['dominant_colors'][:8] # до 8 цветов
            for i, color in enumerate(palette_colors):
                fig.add_shape(type="rect",
                    x0=i, y0=0, x1=i+0.9, y1=1,
                    line=dict(width=0),
                    fillcolor=f'rgb({color[0]},{color[1]},{color[2]})',
                    row=2, col=1
                )
            fig.update_xaxes(showticklabels=False, row=2, col=1)
            fig.update_yaxes(showticklabels=False, range=[0, 1], row=2, col=1)

        fig.update_layout(
            height=600, title_text="Анализ цветовых характеристик",
            template=self.plot_config['template']
        )
        return fig
    
    def plot_composition_analysis(self, composition_data: Dict[str, Any]) -> go.Figure:
        """Создание графика анализа композиции."""
        # ИСПРАВЛЕНИЕ: Заменен тип 'radar' на 'polar'
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Оценки композиции', 'Визуальный баланс', 'Сравнение с эталонами'],
            specs=[[{'type': 'bar', 'rowspan': 2}, {'type': 'bar'}],
                   [None, {'type': 'polar'}]] # <-- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
        )
        
        # 1. Оценки композиции (вертикальный бар)
        metrics = ['rule_of_thirds_score', 'visual_balance_score', 'composition_complexity', 'center_focus_score', 'symmetry_score']
        metric_names = ['Правило третей', 'Баланс', 'Сложность (инверт.)', 'Фокус', 'Симметрия']
        values = [
            composition_data.get('rule_of_thirds_score', 0),
            composition_data.get('visual_balance_score', 0),
            1 - composition_data.get('composition_complexity', 0), # Инвертируем сложность
            composition_data.get('center_focus_score', 0),
            composition_data.get('symmetry_score', 0)
        ]
        
        fig.add_trace(go.Bar(
            x=values, y=metric_names, orientation='h',
            marker_color=self.color_scheme['primary'], name='Оценки', showlegend=False
        ), row=1, col=1)

        # 2. Визуальный баланс
        balance_score = composition_data.get('visual_balance_score', 0.5)
        # Более интуитивное представление баланса
        left_weight = 1.0 
        right_weight = (2 * balance_score) - 1 if balance_score > 0.5 else 1 - (2 * (0.5 - balance_score))

        fig.add_trace(go.Bar(
            x=['Левая сторона', 'Правая сторона'], y=[left_weight, right_weight],
            marker_color=[self.color_scheme['secondary'], self.color_scheme['secondary']],
            name='Баланс', showlegend=False
        ), row=1, col=2)
        fig.update_yaxes(range=[0, 1], row=1, col=2)

        # 3. Радарная диаграмма сравнения с эталонами
        current_values = [
            composition_data.get('rule_of_thirds_score', 0),
            composition_data.get('visual_balance_score', 0),
            composition_data.get('symmetry_score', 0),
            composition_data.get('center_focus_score', 0),
            1 - composition_data.get('composition_complexity', 0)
        ]
        ideal_values = [0.8, 0.75, 0.6, 0.7, 0.8]
        categories = ['Правило третей', 'Баланс', 'Симметрия', 'Фокус', 'Простота']
        
        fig.add_trace(go.Scatterpolar(
            r=current_values, theta=categories, fill='toself',
            name='Текущий креатив', line_color=self.color_scheme['primary']
        ), row=2, col=2)
        
        fig.add_trace(go.Scatterpolar(
            r=ideal_values, theta=categories, fill='toself',
            name='Эталон', line_color=self.color_scheme['success'], opacity=0.5
        ), row=2, col=2)
        
        fig.update_layout(
            height=600, title_text="Анализ композиционных характеристик",
            template=self.plot_config['template'], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    
    def plot_performance_prediction(self, predictions: Dict[str, float], 
                                  confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None) -> go.Figure:
        """Создание графика предсказания эффективности."""
        metrics = list(predictions.keys())
        values = [v * 100 for v in predictions.values()]
        
        metric_names = {'ctr': 'CTR (%)', 'conversion_rate': 'Конверсия (%)', 'engagement': 'Вовлеченность (%)'}
        display_names = [metric_names.get(m, m) for m in metrics]

        error_y_upper, error_y_lower = [], []
        if confidence_intervals:
            for metric in metrics:
                low, high = confidence_intervals.get(metric, (0,0))
                pred = predictions[metric]
                error_y_upper.append((high - pred) * 100)
                error_y_lower.append((pred - low) * 100)

        fig = go.Figure(data=go.Bar(
            x=display_names, y=values,
            marker_color=self.color_scheme['primary'],
            text=[f'{v:.2f}%' for v in values], textposition='auto',
            error_y=dict(type='data', array=error_y_upper, arrayminus=error_y_lower, visible=True) if confidence_intervals else None
        ))
        
        fig.update_layout(
            title="Предсказание эффективности креатива (с доверительными интервалами)",
            xaxis_title="Метрики эффективности", yaxis_title="Значение (%)",
            template=self.plot_config['template'], height=self.plot_config['height']
        )
        return fig
    
    def plot_feature_importance(self, feature_importance: List[Tuple[str, float]], target: str = 'CTR') -> go.Figure:
        """Создание графика важности признаков."""
        if not feature_importance:
            fig = go.Figure()
            fig.add_annotation(text="Нет данных о важности признаков", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        features, importance = zip(*feature_importance)
        
        feature_translations = {
            'brightness': 'Яркость', 'saturation': 'Насыщенность', 'contrast_score': 'Контрастность',
            'color_temperature': 'Цвет. температура', 'harmony_score': 'Гармония', 'rule_of_thirds_score': 'Правило третей',
            'visual_balance_score': 'Баланс', 'composition_complexity': 'Сложность', 'text_contrast': 'Контраст текста',
            'readability_score': 'Читаемость', 'has_cta': 'Наличие CTA', 'center_focus_score': 'Фокус',
            'text_hierarchy': 'Иерархия текста', 'cat_E-commerce': 'Кат: E-commerce', 'cat_Финансы': 'Кат: Финансы',
            'cat_Автомобили': 'Кат: Авто', 'reg_Россия': 'Регион: РФ', 'reg_США': 'Регион: США'
        }
        translated_features = [feature_translations.get(f, f) for f in features]
        
        fig = go.Figure(data=go.Bar(
            x=importance, y=translated_features, orientation='h',
            marker_color=self.color_scheme['primary']
        ))
        
        fig.update_layout(
            title=f"Ключевые факторы для предсказания {target}",
            xaxis_title="Важность фактора", yaxis_title="Факторы",
            template=self.plot_config['template'], height=max(400, len(features) * 30),
            yaxis=dict(autorange="reversed"), margin=dict(l=150)
        )
        return fig
