# visualizer.py
"""
Модуль визуализации для Creative Performance Predictor.
Создает интерактивные графики и диаграммы для анализа креативов.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2

from config import COLOR_SCHEME, PLOT_CONFIG, get_color_name, format_percentage

class Visualizer:
    """
    Класс для создания визуализаций анализа креативов.
    Включает графики анализа, предсказаний и рекомендаций.
    """
    
    def __init__(self):
        self.color_scheme = COLOR_SCHEME
        self.plot_config = PLOT_CONFIG
        
    def plot_color_analysis(self, color_data: Dict[str, Any]) -> go.Figure:
        """
        Создание графика анализа цветовых характеристик.
        
        Args:
            color_data: Результаты цветового анализа
            
        Returns:
            go.Figure: Plotly график цветового анализа
        """
        # Создание subplot с несколькими графиками
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Доминирующие цвета', 'Цветовые метрики',
                'Цветовая палитра', 'Распределение характеристик'
            ],
            specs=[[{'type': 'bar'}, {'type': 'indicator'}],
                   [{'type': 'scatter'}, {'type': 'polar'}]]
        )
        
        # 1. Доминирующие цвета
        if 'dominant_colors' in color_data:
            colors = color_data['dominant_colors'][:5]  # Топ-5 цветов
            color_names = [get_color_name(color) for color in colors]
            color_hex = [f'rgb({color[0]},{color[1]},{color[2]})' for color in colors]
            
            fig.add_trace(
                go.Bar(
                    x=color_names,
                    y=[1] * len(colors),
                    marker_color=color_hex,
                    name='Доминирующие цвета',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Цветовые метрики
        metrics = ['brightness', 'saturation', 'contrast_score', 'harmony_score']
        metric_values = [color_data.get(metric, 0) for metric in metrics]
        metric_names = ['Яркость', 'Насыщенность', 'Контрастность', 'Гармония']
        
        for i, (name, value) in enumerate(zip(metric_names, metric_values)):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    domain={'x': [0.5 + (i%2)*0.25, 0.75 + (i%2)*0.25], 
                           'y': [0.7 - (i//2)*0.3, 1.0 - (i//2)*0.3]},
                    title={'text': name},
                    gauge={
                        'axis': {'range': [None, 1]},
                        'bar': {'color': self.color_scheme['primary']},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 1], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.8
                        }
                    }
                ),
                row=1, col=2
            )
        
        # 3. Цветовая температура vs Насыщенность
        fig.add_trace(
            go.Scatter(
                x=[color_data.get('color_temperature', 0.5)],
                y=[color_data.get('saturation', 0.5)],
                mode='markers',
                marker=dict(
                    size=20,
                    color=self.color_scheme['primary'],
                    symbol='circle'
                ),
                name='Ваш креатив',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Добавление областей оптимальности
        fig.add_shape(
            type="rect",
            x0=0.4, y0=0.4, x1=0.8, y1=0.8,
            line=dict(color="green", dash="dash"),
            fillcolor="lightgreen",
            opacity=0.2,
            row=2, col=1
        )
        
        # 4. Радарная диаграмма характеристик
        radar_metrics = ['Яркость', 'Насыщенность', 'Контрастность', 'Гармония', 'Разнообразие']
        radar_values = [
            color_data.get('brightness', 0),
            color_data.get('saturation', 0),
            color_data.get('contrast_score', 0),
            color_data.get('harmony_score', 0),
            color_data.get('color_diversity', 0) / 10.0  # Нормализация
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=radar_values,
                theta=radar_metrics,
                fill='toself',
                name='Цветовой профиль',
                line_color=self.color_scheme['primary']
            ),
            row=2, col=2
        )
        
        # Обновление layout
        fig.update_layout(
            height=600,
            title_text="Анализ цветовых характеристик креатива",
            template=self.plot_config['template']
        )
        
        # Обновление осей
        fig.update_xaxes(title_text="Цветовая температура (холодная ← → теплая)", row=2, col=1)
        fig.update_yaxes(title_text="Насыщенность", row=2, col=1)
        
        return fig
    
    def plot_composition_analysis(self, composition_data: Dict[str, Any]) -> go.Figure:
        """
        Создание графика анализа композиции.
        
        Args:
            composition_data: Результаты композиционного анализа
            
        Returns:
            go.Figure: Plotly график композиционного анализа
        """
        # Создание subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Оценки композиции', 'Правило третей',
                'Визуальный баланс', 'Сравнение с эталонами'
            ],
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'radar'}]]
        )
        
        # 1. Оценки композиции
        metrics = [
            'rule_of_thirds_score', 'visual_balance_score', 
            'composition_complexity', 'center_focus_score'
        ]
        metric_names = [
            'Правило третей', 'Визуальный баланс',
            'Сложность', 'Центральный фокус'
        ]
        values = [composition_data.get(metric, 0) for metric in metrics]
        
        colors = [self.color_scheme['success'] if v > 0.6 else 
                 self.color_scheme['warning'] if v > 0.3 else 
                 self.color_scheme['warning'] for v in values]
        
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=values,
                marker_color=colors,
                name='Оценки',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Визуализация правила третей
        # Создаем сетку правила третей
        grid_x = [0.33, 0.33, 0.67, 0.67, 0, 1, 0, 1]
        grid_y = [0, 1, 0, 1, 0.33, 0.33, 0.67, 0.67]
        
        fig.add_trace(
            go.Scatter(
                x=grid_x[:4],
                y=grid_y[:4],
                mode='lines',
                line=dict(color='lightblue', dash='dash'),
                name='Сетка третей',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Точки пересечения
        intersection_x = [0.33, 0.67, 0.33, 0.67]
        intersection_y = [0.33, 0.33, 0.67, 0.67]
        
        fig.add_trace(
            go.Scatter(
                x=intersection_x,
                y=intersection_y,
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.color_scheme['primary'],
                    symbol='x'
                ),
                name='Точки силы',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Визуальный баланс (левая vs правая сторона)
        balance_score = composition_data.get('visual_balance_score', 0.5)
        left_weight = balance_score
        right_weight = 1 - balance_score
        
        fig.add_trace(
            go.Bar(
                x=['Левая сторона', 'Правая сторона'],
                y=[left_weight, right_weight],
                marker_color=[self.color_scheme['primary'], self.color_scheme['secondary']],
                name='Баланс',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Радарная диаграмма сравнения с эталонами
        current_values = [
            composition_data.get('rule_of_thirds_score', 0),
            composition_data.get('visual_balance_score', 0),
            composition_data.get('symmetry_score', 0),
            composition_data.get('center_focus_score', 0),
            1 - composition_data.get('composition_complexity', 0.5)  # Инвертируем сложность
        ]
        
        ideal_values = [0.8, 0.7, 0.6, 0.7, 0.8]  # Эталонные значения
        
        categories = ['Правило третей', 'Баланс', 'Симметрия', 'Фокус', 'Простота']
        
        fig.add_trace(
            go.Scatterpolar(
                r=current_values,
                theta=categories,
                fill='toself',
                name='Текущий креатив',
                line_color=self.color_scheme['primary']
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatterpolar(
                r=ideal_values,
                theta=categories,
                fill='toself',
                name='Эталон',
                line_color=self.color_scheme['success'],
                opacity=0.6
            ),
            row=2, col=2
        )
        
        # Обновление layout
        fig.update_layout(
            height=600,
            title_text="Анализ композиционных характеристик",
            template=self.plot_config['template']
        )
        
        return fig
    
    def plot_performance_prediction(self, predictions: Dict[str, float], 
                                  confidence_intervals: Dict[str, Tuple[float, float]] = None) -> go.Figure:
        """
        Создание графика предсказания эффективности.
        
        Args:
            predictions: Предсказанные метрики
            confidence_intervals: Доверительные интервалы
            
        Returns:
            go.Figure: Plotly график предсказаний
        """
        metrics = list(predictions.keys())
        values = list(predictions.values())
        
        # Преобразование метрик в проценты для лучшего восприятия
        if 'ctr' in metrics:
            idx = metrics.index('ctr')
            values[idx] = values[idx] * 100
            metrics[idx] = 'CTR (%)'
        
        if 'conversion_rate' in metrics:
            idx = metrics.index('conversion_rate')
            values[idx] = values[idx] * 100
            metrics[idx] = 'Conversion Rate (%)'
        
        if 'engagement' in metrics:
            idx = metrics.index('engagement')
            values[idx] = values[idx] * 100
            metrics[idx] = 'Engagement (%)'
        
        # Определение цветов на основе производительности
        colors = []
        for i, value in enumerate(values):
            if metrics[i] == 'CTR (%)':
                threshold = 2.0  # 2% CTR
            elif metrics[i] == 'Conversion Rate (%)':
                threshold = 5.0  # 5% Conversion
            else:
                threshold = 10.0  # 10% Engagement
            
            if value >= threshold:
                colors.append(self.color_scheme['success'])
            elif value >= threshold * 0.7:
                colors.append(self.color_scheme['warning'])
            else:
                colors.append(self.color_scheme['warning'])
        
        # Создание основного графика
        fig = go.Figure()
        
        # Основные столбцы
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                marker_color=colors,
                name='Предсказанная эффективность',
                text=[f'{v:.2f}' for v in values],
                textposition='auto'
            )
        )
        
        # Добавление доверительных интервалов если есть
        if confidence_intervals:
            error_y = []
            for i, metric in enumerate(['ctr', 'conversion_rate', 'engagement']):
                if metric in confidence_intervals:
                    lower, upper = confidence_intervals[metric]
                    # Преобразование в проценты
                    if metric == 'ctr' or metric == 'conversion_rate':
                        lower *= 100
                        upper *= 100
                    elif metric == 'engagement':
                        lower *= 100
                        upper *= 100
                    
                    error_y.append({
                        'type': 'data',
                        'symmetric': False,
                        'array': [upper - values[i]],
                        'arrayminus': [values[i] - lower]
                    })
            
            if error_y:
                fig.update_traces(error_y=error_y[0])
        
        # Добавление целевых линий
        target_lines = {
            'CTR (%)': 2.0,
            'Conversion Rate (%)': 5.0,
            'Engagement (%)': 10.0
        }
        
        for metric, target in target_lines.items():
            if metric in metrics:
                fig.add_hline(
                    y=target,
                    line_dash="dash",
                    line_color=self.color_scheme['info'],
                    annotation_text=f"Цель: {target}%"
                )
        
        fig.update_layout(
            title="Предсказание эффективности креатива",
            xaxis_title="Метрики эффективности",
            yaxis_title="Значение (%)",
            template=self.plot_config['template'],
            height=self.plot_config['height']
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance: List[Tuple[str, float]], 
                               target: str = 'CTR') -> go.Figure:
        """
        Создание графика важности признаков.
        
        Args:
            feature_importance: Список кортежей (признак, важность)
            target: Целевая метрика
            
        Returns:
            go.Figure: Plotly график важности признаков
        """
        if not feature_importance:
            # Создание пустого графика
            fig = go.Figure()
            fig.add_annotation(
                text="Нет данных о важности признаков",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle'
            )
            return fig
        
        # Сортировка по важности
        features, importance = zip(*feature_importance)
        
        # Перевод названий признаков на русский
        feature_translations = {
            'brightness': 'Яркость',
            'saturation': 'Насыщенность',
            'contrast_score': 'Контрастность',
            'color_temperature': 'Цветовая температура',
            'harmony_score': 'Цветовая гармония',
            'rule_of_thirds_score': 'Правило третей',
            'visual_balance_score': 'Визуальный баланс',
            'composition_complexity': 'Сложность композиции',
            'text_contrast': 'Контрастность текста',
            'readability_score': 'Читаемость',
            'has_cta': 'Наличие CTA',
            'center_focus_score': 'Центральный фокус',
            'text_hierarchy': 'Иерархия текста'
        }
        
        translated_features = [
            feature_translations.get(feature, feature) for feature in features
        ]
        
        # Создание горизонтального bar chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=importance,
                y=translated_features,
                orientation='h',
                marker_color=self.color_scheme['primary'],
                text=[f'{imp:.3f}' for imp in importance],
                textposition='auto'
            )
        )
        
        fig.update_layout(
            title=f"Важность признаков для предсказания {target}",
            xaxis_title="Важность признака",
            yaxis_title="Признаки",
            template=self.plot_config['template'],
            height=max(400, len(features) * 25),
            margin=dict(l=150)  # Больше места для названий признаков
        )
        
        return fig
    
    def plot_image_analysis_overlay(self, image: Image.Image, 
                                   analysis_data: Dict[str, Any]) -> Image.Image:
        """
        Создание наложения анализа на исходное изображение.
        
        Args:
            image: Исходное изображение
            analysis_data: Результаты анализа изображения
            
        Returns:
            Image.Image: Изображение с наложенным анализом
        """
        # Создание копии изображения
        overlay_image = image.copy()
        draw = ImageDraw.Draw(overlay_image)
        
        width, height = image.size
        
        # Рисование сетки правила третей
        if analysis_data.get('show_rule_of_thirds', True):
            # Вертикальные линии
            line_color = (255, 255, 255, 128)  # Белый с прозрачностью
            line_width = 2
            
            v1 = width // 3
            v2 = 2 * width // 3
            h1 = height // 3
            h2 = 2 * height // 3
            
            # Вертикальные линии
            draw.line([(v1, 0), (v1, height)], fill=line_color[:3], width=line_width)
            draw.line([(v2, 0), (v2, height)], fill=line_color[:3], width=line_width)
            
            # Горизонтальные линии
            draw.line([(0, h1), (width, h1)], fill=line_color[:3], width=line_width)
            draw.line([(0, h2), (width, h2)], fill=line_color[:3], width=line_width)
            
            # Точки пересечения (точки силы)
            point_size = 8
            point_color = (255, 0, 0)  # Красный
            
            for x in [v1, v2]:
                for y in [h1, h2]:
                    draw.ellipse(
                        [x-point_size, y-point_size, x+point_size, y+point_size],
                        fill=point_color
                    )
        
        # Добавление текстовой информации
        if analysis_data.get('show_metrics', True):
            try:
                # Попытка использовать системный шрифт
                font_size = max(16, min(width, height) // 40)
                font = ImageFont.load_default()
            except:
                font = None
            
            # Фон для текста
            text_bg_color = (0, 0, 0, 180)  # Черный с прозрачностью
            text_color = (255, 255, 255)   # Белый текст
            
            # Подготовка текста с метриками
            metrics_text = []
            
            if 'color_analysis' in analysis_data:
                color_data = analysis_data['color_analysis']
                metrics_text.extend([
                    f"Яркость: {color_data.get('brightness', 0):.2f}",
                    f"Контрастность: {color_data.get('contrast_score', 0):.2f}",
                    f"Гармония: {color_data.get('harmony_score', 0):.2f}"
                ])
            
            if 'composition_analysis' in analysis_data:
                comp_data = analysis_data['composition_analysis']
                metrics_text.extend([
                    f"Правило третей: {comp_data.get('rule_of_thirds_score', 0):.2f}",
                    f"Баланс: {comp_data.get('visual_balance_score', 0):.2f}"
                ])
            
            # Рисование текста в углу
            text_x = 10
            text_y = 10
            line_height = 25
            
            for i, text in enumerate(metrics_text[:5]):  # Максимум 5 строк
                y_pos = text_y + i * line_height
                
                # Фон для текста
                text_bbox = draw.textbbox((text_x, y_pos), text, font=font)
                draw.rectangle(
                    [text_bbox[0]-5, text_bbox[1]-2, text_bbox[2]+5, text_bbox[3]+2],
                    fill=text_bg_color[:3]
                )
                
                # Сам текст
                draw.text((text_x, y_pos), text, fill=text_color, font=font)
        
        return overlay_image
    
    def create_performance_dashboard(self, analysis_results: Dict[str, Any]) -> List[go.Figure]:
        """
        Создание полного дашборда с результатами анализа.
        
        Args:
            analysis_results: Полные результаты анализа креатива
            
        Returns:
            List[go.Figure]: Список графиков для дашборда
        """
        figures = []
        
        # 1. График цветового анализа
        if 'color_analysis' in analysis_results:
            color_fig = self.plot_color_analysis(analysis_results['color_analysis'])
            figures.append(color_fig)
        
        # 2. График композиционного анализа
        if 'composition_analysis' in analysis_results:
            comp_fig = self.plot_composition_analysis(analysis_results['composition_analysis'])
            figures.append(comp_fig)
        
        # 3. График предсказаний
        if 'predictions' in analysis_results:
            pred_fig = self.plot_performance_prediction(
                analysis_results['predictions'],
                analysis_results.get('confidence_intervals')
            )
            figures.append(pred_fig)
        
        # 4. График важности признаков
        if 'feature_importance' in analysis_results:
            importance_fig = self.plot_feature_importance(
                analysis_results['feature_importance']
            )
            figures.append(importance_fig)
        
        return figures
    
    def create_comparison_chart(self, current_metrics: Dict[str, float], 
                               benchmark_metrics: Dict[str, float]) -> go.Figure:
        """
        Создание графика сравнения с бенчмарками.
        
        Args:
            current_metrics: Текущие метрики креатива
            benchmark_metrics: Бенчмарк метрики
            
        Returns:
            go.Figure: График сравнения
        """
        metrics = list(current_metrics.keys())
        current_values = [current_metrics[m] * 100 for m in metrics]  # В процентах
        benchmark_values = [benchmark_metrics.get(m, 0) * 100 for m in metrics]
        
        # Перевод названий метрик
        metric_names = {
            'ctr': 'CTR (%)',
            'conversion_rate': 'Conversion Rate (%)',
            'engagement': 'Engagement (%)'
        }
        
        translated_metrics = [metric_names.get(m, m) for m in metrics]
        
        fig = go.Figure()
        
        # Текущие значения
        fig.add_trace(
            go.Bar(
                name='Ваш креатив',
                x=translated_metrics,
                y=current_values,
                marker_color=self.color_scheme['primary']
            )
        )
        
        # Бенчмарк значения
        fig.add_trace(
            go.Bar(
                name='Средний по отрасли',
                x=translated_metrics,
                y=benchmark_values,
                marker_color=self.color_scheme['secondary']
            )
        )
        
        fig.update_layout(
            title='Сравнение с отраслевыми бенчмарками',
            xaxis_title='Метрики',
            yaxis_title='Значение (%)',
            barmode='group',
            template=self.plot_config['template'],
            height=self.plot_config['height']
        )
        
        return fig