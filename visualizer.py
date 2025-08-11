# visualizer.py - РЕВОЛЮЦИОННАЯ ВЕРСИЯ
"""
Модуль визуализации для Creative Performance Predictor.
Продвинутые интерактивные графики, heatmaps и 3D визуализации.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import colorsys
import math

from config import COLOR_SCHEME, PLOT_CONFIG, get_color_name

class AdvancedVisualizer:
    """
    Продвинутый класс для создания интерактивных визуализаций анализа креативов.
    Включает heatmaps, 3D графики, анимации и научно обоснованные диаграммы.
    """
    
    def __init__(self):
        self.color_scheme = COLOR_SCHEME
        self.plot_config = PLOT_CONFIG
        
        # Расширенная цветовая палитра
        self.advanced_colors = {
            'performance_excellent': '#00C851',
            'performance_good': '#33B679', 
            'performance_average': '#FF9800',
            'performance_poor': '#F44336',
            'ctr_color': '#2196F3',
            'conversion_color': '#4CAF50', 
            'engagement_color': '#FF9800',
            'attention_heat': '#FF5722',
            'trust_color': '#3F51B5',
            'emotion_color': '#E91E63'
        }
        
        # Научно обоснованные пороги
        self.performance_thresholds = {
            'ctr': {'excellent': 0.04, 'good': 0.025, 'average': 0.015},
            'conversion_rate': {'excellent': 0.08, 'good': 0.05, 'average': 0.03},
            'engagement': {'excellent': 0.15, 'good': 0.10, 'average': 0.06}
        }
        
    def create_performance_dashboard(self, predictions: Dict[str, float], 
                                   confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
                                   benchmarks: Optional[Dict[str, float]] = None) -> go.Figure:
        """Создание интерактивного дашборда эффективности."""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Прогнозы vs Бенчмарки', 'Рейтинг эффективности', 'Доверительные интервалы',
                'Композитный индекс', 'Тренд анализ', 'ROI потенциал'
            ],
            specs=[
                [{'type': 'bar'}, {'type': 'indicator'}, {'type': 'scatter'}],
                [{'type': 'scatterpolar'}, {'type': 'scatter'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        metrics = ['CTR', 'Конверсия', 'Вовлеченность']
        values = [predictions['ctr'] * 100, predictions['conversion_rate'] * 100, predictions['engagement'] * 100]
        colors = [self.advanced_colors['ctr_color'], self.advanced_colors['conversion_color'], self.advanced_colors['engagement_color']]
        
        # 1. Прогнозы vs Бенчмарки
        if benchmarks:
            benchmark_values = [benchmarks.get('ctr', 0.02) * 100, 
                              benchmarks.get('conversion_rate', 0.05) * 100,
                              benchmarks.get('engagement', 0.1) * 100]
            
            fig.add_trace(go.Bar(
                x=metrics, y=benchmark_values, name='Бенчмарк отрасли',
                marker_color='rgba(128,128,128,0.5)', offsetgroup=1
            ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=metrics, y=values, name='Ваши прогнозы',
            marker_color=colors, offsetgroup=2,
            text=[f'{v:.2f}%' for v in values], textposition='auto'
        ), row=1, col=1)
        
        # 2. Композитный индекс эффективности
        composite_score = np.mean([
            values[0] / (self.performance_thresholds['ctr']['good'] * 100),
            values[1] / (self.performance_thresholds['conversion_rate']['good'] * 100),
            values[2] / (self.performance_thresholds['engagement']['good'] * 100)
        ]) * 100
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=composite_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Общая оценка"},
            delta={'reference': 100},
            gauge={
                'axis': {'range': [None, 200]},
                'bar': {'color': self._get_performance_color(composite_score)},
                'steps': [
                    {'range': [0, 80], 'color': "lightgray"},
                    {'range': [80, 120], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 'value': 100}
            }
        ), row=1, col=2)
        
        # 3. Доверительные интервалы
        if confidence_intervals:
            x_pos = [1, 2, 3]
            for i, (metric, (lower, upper)) in enumerate(confidence_intervals.items()):
                y_val = predictions[metric] * 100
                error_y = (upper - predictions[metric]) * 100
                error_y_minus = (predictions[metric] - lower) * 100
                
                fig.add_trace(go.Scatter(
                    x=[x_pos[i]], y=[y_val],
                    error_y=dict(type='data', array=[error_y], arrayminus=[error_y_minus]),
                    mode='markers', marker_size=12, marker_color=colors[i],
                    name=f'{metrics[i]} ±95%', showlegend=False
                ), row=1, col=3)
        
        # 4. Радарная диаграмма детальных метрик
        detailed_metrics = ['Привлекательность', 'Читаемость', 'Доверие', 'Эмоциональность', 'Профессионализм']
        detailed_values = [
            composite_score * 0.8,  # Примерные значения на основе общей оценки
            composite_score * 0.9,
            composite_score * 0.7,
            composite_score * 0.85,
            composite_score * 0.75
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=detailed_values, theta=detailed_metrics, fill='toself',
            name='Детальная оценка', line_color=self.advanced_colors['trust_color']
        ), row=2, col=1)
        
        # 5. Тренд потенциала (симулированный)
        months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн']
        baseline = [composite_score] * 6
        with_improvements = [composite_score + i * 5 for i in range(6)]
        
        fig.add_trace(go.Scatter(
            x=months, y=baseline, mode='lines', name='Текущий уровень',
            line=dict(dash='dash', color='gray')
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=months, y=with_improvements, mode='lines+markers', name='С улучшениями',
            line=dict(color=self.advanced_colors['performance_excellent'])
        ), row=2, col=2)
        
        # 6. ROI потенциал по метрикам
        roi_potential = [
            (values[0] - self.performance_thresholds['ctr']['average'] * 100) * 2,
            (values[1] - self.performance_thresholds['conversion_rate']['average'] * 100) * 3,
            (values[2] - self.performance_thresholds['engagement']['average'] * 100) * 1.5
        ]
        
        fig.add_trace(go.Bar(
            x=metrics, y=roi_potential, name='ROI потенциал',
            marker_color=[self._get_roi_color(roi) for roi in roi_potential],
            text=[f'+{roi:.1f}%' if roi > 0 else f'{roi:.1f}%' for roi in roi_potential],
            textposition='auto'
        ), row=2, col=3)
        
        fig.update_layout(
            height=800, title_text="📊 Интеллектуальный дашборд эффективности креатива",
            template=self.plot_config['template'], showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_attention_heatmap(self, image_features: Dict, predictions: Dict) -> go.Figure:
        """Создание heatmap зон внимания на основе композиционного анализа."""
        
        # Симулируем тепловую карту на основе характеристик изображения
        grid_size = 20
        x = np.linspace(0, 100, grid_size)
        y = np.linspace(0, 100, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Создаем базовую карту внимания
        attention_map = np.zeros((grid_size, grid_size))
        
        # Правило третей - горячие точки
        third_points_x = [33, 67]
        third_points_y = [33, 67]
        
        for tx in third_points_x:
            for ty in third_points_y:
                for i in range(grid_size):
                    for j in range(grid_size):
                        dist = np.sqrt((X[i,j] - tx)**2 + (Y[i,j] - ty)**2)
                        attention_map[i,j] += np.exp(-dist/15) * 0.8
        
        # Центральный фокус
        center_strength = image_features.get('center_focus_score', 0.5)
        for i in range(grid_size):
            for j in range(grid_size):
                dist_center = np.sqrt((X[i,j] - 50)**2 + (Y[i,j] - 50)**2)
                attention_map[i,j] += np.exp(-dist_center/20) * center_strength * 0.6
        
        # Контраст влияет на привлечение внимания
        contrast_boost = image_features.get('contrast_score', 0.5)
        attention_map *= (1 + contrast_boost)
        
        # Текст привлекает внимание
        if image_features.get('has_cta', False):
            # Предполагаем CTA в правом нижнем углу
            for i in range(int(grid_size*0.7), grid_size):
                for j in range(int(grid_size*0.7), grid_size):
                    attention_map[i,j] += 0.4
        
        # Нормализация
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # ИСПРАВЛЕННАЯ ВЕРСИЯ с правильными параметрами colorbar
        fig = go.Figure(data=go.Heatmap(
            z=attention_map, 
            x=x, 
            y=y,
            colorscale=[
                [0, 'rgba(255,255,255,0)'],
                [0.3, 'rgba(255,255,0,0.3)'],
                [0.6, 'rgba(255,165,0,0.6)'],
                [1, 'rgba(255,0,0,0.9)']
            ],
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Интенсивность внимания",
                    side="right"
                ),
                orientation="v",
                len=0.9
            )
        ))
        
        # Добавляем сетку правила третей
        fig.add_hline(y=33.33, line_dash="dash", line_color="white", opacity=0.7)
        fig.add_hline(y=66.67, line_dash="dash", line_color="white", opacity=0.7)
        fig.add_vline(x=33.33, line_dash="dash", line_color="white", opacity=0.7)
        fig.add_vline(x=66.67, line_dash="dash", line_color="white", opacity=0.7)
        
        # Добавляем точки силы
        fig.add_trace(go.Scatter(
            x=[33.33, 66.67, 33.33, 66.67], 
            y=[33.33, 33.33, 66.67, 66.67],
            mode='markers', 
            marker=dict(size=12, color='white', symbol='x'),
            name='Точки силы', 
            showlegend=True
        ))
        
        fig.update_layout(
            title="🎯 Карта внимания пользователей",
            xaxis_title="Горизонтальная позиция (%)",
            yaxis_title="Вертикальная позиция (%)",
            height=500, 
            template=self.plot_config['template'],
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100], scaleanchor="x", scaleratio=1)
        )
        
        return fig
    
    def create_color_psychology_analysis(self, color_data: Dict[str, Any]) -> go.Figure:
        """Создание анализа цветовой психологии."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Эмоциональное воздействие', 'Доминирующие цвета', 'Цветовые ассоциации', 'Оптимизация палитры'],
            specs=[
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ]
        )
        
        # 1. Эмоциональное воздействие цветов
        emotions = ['Доверие', 'Энергия', 'Спокойствие', 'Теплота', 'Профессионализм']
        
        # Рассчитываем эмоциональное воздействие на основе цветов
        emotion_scores = self._calculate_emotion_scores(color_data)
        
        fig.add_trace(go.Bar(
            x=emotions, y=emotion_scores,
            marker_color=[
                self.advanced_colors['trust_color'],
                self.advanced_colors['attention_heat'],
                '#4CAF50', '#FF9800', '#3F51B5'
            ],
            text=[f'{score:.1f}' for score in emotion_scores],
            textposition='auto'
        ), row=1, col=1)
        
        # 2. Распределение доминирующих цветов
        if 'dominant_colors' in color_data and color_data['dominant_colors']:
            colors = color_data['dominant_colors'][:5]
            color_names = [self._get_color_emotion(color) for color in colors]
            color_hex = [f'rgb({c[0]},{c[1]},{c[2]})' for c in colors]
            
            fig.add_trace(go.Pie(
                labels=color_names, values=[1]*len(colors),
                marker_colors=color_hex,
                textinfo='label+percent'
            ), row=1, col=2)
        
        # 3. Цветовые ассоциации по категориям
        categories = ['Продажи', 'Доверие', 'Премиум', 'Молодежь', 'Экология']
        current_fit = [
            self._assess_category_fit(color_data, cat) for cat in categories
        ]
        optimal_fit = [0.9, 0.85, 0.7, 0.6, 0.8]  # Оптимальные значения
        
        fig.add_trace(go.Scatter(
            x=categories, y=current_fit, mode='markers+lines',
            marker=dict(size=12, color=self.advanced_colors['ctr_color']),
            name='Текущее соответствие'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=categories, y=optimal_fit, mode='markers+lines',
            marker=dict(size=8, color=self.advanced_colors['performance_excellent']),
            line=dict(dash='dash'), name='Оптимальное'
        ), row=2, col=1)
        
        # 4. Рекомендации по оптимизации
        optimization_areas = ['Контраст', 'Гармония', 'Насыщенность', 'Температура']
        current_values = [
            color_data.get('contrast_score', 0.5) * 100,
            color_data.get('harmony_score', 0.5) * 100,
            color_data.get('saturation', 0.5) * 100,
            color_data.get('color_temperature', 0.5) * 100
        ]
        target_values = [80, 75, 70, 60]  # Оптимальные значения
        
        fig.add_trace(go.Bar(
            x=optimization_areas, y=current_values, name='Текущие',
            marker_color=self.advanced_colors['conversion_color'], opacity=0.7
        ), row=2, col=2)
        
        fig.add_trace(go.Bar(
            x=optimization_areas, y=target_values, name='Цель',
            marker_color=self.advanced_colors['performance_excellent'], opacity=0.5
        ), row=2, col=2)
        
        fig.update_layout(
            height=700, title_text="🎨 Анализ цветовой психологии и эмоционального воздействия",
            template=self.plot_config['template']
        )
        
        return fig
    
    def create_performance_prediction_detailed(self, predictions: Dict[str, float],
                                             feature_importance: List[Tuple[str, float]],
                                             confidence_intervals: Dict) -> go.Figure:
        """Детальный анализ предсказаний с декомпозицией факторов."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Вклад факторов в CTR', 'Анализ неопределенности', 'Сравнение с бенчмарками', 'Потенциал роста'],
            specs=[
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'waterfall'}]
            ]
        )
        
        # 1. Декомпозиция важности факторов
        if feature_importance:
            factors = [self._translate_feature_name(feat) for feat, _ in feature_importance[:8]]
            importance_values = [imp * 100 for _, imp in feature_importance[:8]]
            
            # Цветовое кодирование по типу фактора
            colors = [self._get_factor_color(feat) for feat in factors]
            
            fig.add_trace(go.Bar(
                y=factors, x=importance_values, orientation='h',
                marker_color=colors,
                text=[f'{val:.1f}%' for val in importance_values],
                textposition='auto'
            ), row=1, col=1)
        
        # 2. Анализ неопределенности и доверительных интервалов
        metrics = ['CTR', 'Конверсия', 'Вовлеченность']
        pred_values = [predictions['ctr'] * 100, predictions['conversion_rate'] * 100, predictions['engagement'] * 100]
        
        if confidence_intervals:
            lower_bounds = [confidence_intervals['ctr'][0] * 100, 
                          confidence_intervals['conversion_rate'][0] * 100,
                          confidence_intervals['engagement'][0] * 100]
            upper_bounds = [confidence_intervals['ctr'][1] * 100,
                          confidence_intervals['conversion_rate'][1] * 100, 
                          confidence_intervals['engagement'][1] * 100]
            
            for i, metric in enumerate(metrics):
                # Основное предсказание
                fig.add_trace(go.Scatter(
                    x=[metric], y=[pred_values[i]], 
                    mode='markers', marker=dict(size=15, color=self.advanced_colors['ctr_color']),
                    name=f'{metric} (прогноз)', showlegend=(i==0)
                ), row=1, col=2)
                
                # Доверительный интервал
                fig.add_trace(go.Scatter(
                    x=[metric, metric], y=[lower_bounds[i], upper_bounds[i]],
                    mode='lines', line=dict(width=6, color=self.advanced_colors['ctr_color']),
                    name='95% интервал', showlegend=(i==0), opacity=0.3
                ), row=1, col=2)
        
        # 3. Сравнение с отраслевыми бенчмарками
        industry_benchmarks = [2.5, 5.0, 10.0]  # Примерные бенчмарки
        
        x_metrics = ['CTR', 'Конверсия', 'Вовлеченность']
        
        fig.add_trace(go.Bar(
            x=x_metrics, y=pred_values, name='Ваш прогноз',
            marker_color=self.advanced_colors['ctr_color']
        ), row=2, col=1)
        
        fig.add_trace(go.Bar(
            x=x_metrics, y=industry_benchmarks, name='Отраслевой бенчмарк',
            marker_color=self.advanced_colors['performance_good'], opacity=0.7
        ), row=2, col=1)
        
        # 4. Waterfall chart потенциала улучшений
        baseline = pred_values[0]  # Берем CTR как базу
        improvements = [
            ('Базовый CTR', baseline),
            ('Цветовая оптимизация', 0.3),
            ('Улучшение композиции', 0.4),
            ('Текстовая оптимизация', 0.2),
            ('Психологические триггеры', 0.3)
        ]
        
        cumulative = baseline
        waterfall_values = [baseline]
        waterfall_labels = ['Текущий']
        
        for label, improvement in improvements[1:]:
            waterfall_values.append(improvement)
            waterfall_labels.append(label)
            cumulative += improvement
        
        waterfall_values.append(cumulative - baseline)
        waterfall_labels.append('Итоговый потенциал')
        
        fig.add_trace(go.Waterfall(
            x=waterfall_labels[1:], y=waterfall_values[1:],
            base=baseline, name="Потенциал улучшений",
            increasing={"marker":{"color":self.advanced_colors['performance_excellent']}},
            totals={"marker":{"color":self.advanced_colors['trust_color']}}
        ), row=2, col=2)
        
        fig.update_layout(
            height=700, title_text="📈 Детальный анализ прогнозов эффективности",
            template=self.plot_config['template']
        )
        
        return fig
    
    def create_recommendation_impact_chart(self, recommendations: List) -> go.Figure:
        """Визуализация влияния рекомендаций."""
        
        if not recommendations:
            fig = go.Figure()
            fig.add_annotation(
                text="Нет рекомендаций для отображения",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font_size=16
            )
            return fig
        
        # Подготовка данных
        titles = [rec.title[:30] + '...' if len(rec.title) > 30 else rec.title for rec in recommendations]
        impacts = [rec.expected_impact * 100 for rec in recommendations]
        efforts = [rec.effort_level for rec in recommendations]
        priorities = [rec.priority for rec in recommendations]
        
        # Размеры пузырьков для effort level
        effort_sizes = {'low': 20, 'medium': 35, 'high': 50}
        sizes = [effort_sizes[effort] for effort in efforts]
        
        # Цвета для приоритетов
        priority_colors = {
            'high': self.advanced_colors['attention_heat'],
            'medium': self.advanced_colors['engagement_color'],
            'low': self.advanced_colors['performance_good']
        }
        colors = [priority_colors[priority] for priority in priorities]
        
        fig = go.Figure()
        
        # Основной scatter plot
        fig.add_trace(go.Scatter(
            x=list(range(len(titles))), y=impacts,
            mode='markers+text',
            marker=dict(size=sizes, color=colors, opacity=0.7,
                       line=dict(width=2, color='white')),
            text=titles, textposition='top center',
            customdata=list(zip(efforts, priorities)),
            hovertemplate='<b>%{text}</b><br>' +
                         'Влияние: %{y:.1f}%<br>' +
                         'Усилия: %{customdata[0]}<br>' +
                         'Приоритет: %{customdata[1]}<br>' +
                         '<extra></extra>'
        ))
        
        # Добавляем зоны приоритетов
        fig.add_hline(y=15, line_dash="dash", line_color="green", 
                     annotation_text="Высокое влияние (>15%)")
        fig.add_hline(y=8, line_dash="dash", line_color="orange",
                     annotation_text="Среднее влияние (8-15%)")
        
        fig.update_layout(
            title="💡 Карта влияния рекомендаций",
            xaxis_title="Рекомендации (упорядочены по приоритету)",
            yaxis_title="Ожидаемое влияние (%)",
            height=500, template=self.plot_config['template'],
            xaxis=dict(showticklabels=False),
            annotations=[
                dict(x=0.02, y=0.98, xref='paper', yref='paper',
                     text='Размер пузырька = уровень усилий', showarrow=False,
                     font=dict(size=12), bgcolor='rgba(255,255,255,0.8)')
            ]
        )
        
        return fig
    
    def create_composition_analysis_3d(self, composition_data: Dict[str, Any]) -> go.Figure:
        """3D анализ композиционных характеристик."""
        
        # Создаем 3D визуализацию композиционных принципов
        metrics = ['rule_of_thirds_score', 'visual_balance_score', 'symmetry_score', 
                  'center_focus_score', 'composition_complexity', 'negative_space']
        
        values = [composition_data.get(metric, 0.5) for metric in metrics]
        
        # Координаты для 3D гистограммы
        x_pos = [0, 1, 2, 0, 1, 2]
        y_pos = [0, 0, 0, 1, 1, 1]
        z_pos = [0] * 6
        
        dx = [0.8] * 6
        dy = [0.8] * 6  
        dz = values
        
        colors = [self._get_composition_color(val) for val in values]
        
        fig = go.Figure(data=[go.Mesh3d(
            x=[0, 1, 2, 0, 1, 2] * 4,  # Создаем 3D столбцы
            y=[0, 0, 0, 1, 1, 1] * 4,
            z=[0, 0, 0, 0, 0, 0] + values + values + [v*2 for v in values],
            colorscale='Viridis',
            intensity=values * 4,
            showscale=True
        )])
        
        # Добавляем подписи
        for i, (metric, value) in enumerate(zip(metrics, values)):
            fig.add_trace(go.Scatter3d(
                x=[x_pos[i]], y=[y_pos[i]], z=[value + 0.1],
                mode='text',
                text=[f'{self._translate_composition_metric(metric)}<br>{value:.2f}'],
                textfont=dict(size=10),
                showlegend=False
            ))
        
        fig.update_layout(
            title="🏗️ 3D Анализ композиции",
            scene=dict(
                xaxis_title="Категория анализа",
                yaxis_title="Тип метрики", 
                zaxis_title="Оценка качества",
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=600
        )
        
        return fig
    
    # === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ===
    
    def _get_performance_color(self, score: float) -> str:
        """Получение цвета на основе оценки производительности."""
        if score >= 120:
            return self.advanced_colors['performance_excellent']
        elif score >= 100:
            return self.advanced_colors['performance_good']
        elif score >= 80:
            return self.advanced_colors['engagement_color']
        else:
            return self.advanced_colors['performance_poor']
    
    def _get_roi_color(self, roi: float) -> str:
        """Цвет на основе ROI потенциала."""
        if roi > 5:
            return self.advanced_colors['performance_excellent']
        elif roi > 0:
            return self.advanced_colors['performance_good']
        else:
            return self.advanced_colors['performance_poor']
    
    def _calculate_emotion_scores(self, color_data: Dict) -> List[float]:
        """Расчет эмоциональных оценок на основе цветов."""
        # Базовые оценки на основе характеристик
        harmony = color_data.get('harmony_score', 0.5)
        temperature = color_data.get('color_temperature', 0.5)
        saturation = color_data.get('saturation', 0.5)
        brightness = color_data.get('brightness', 0.5)
        
        trust = harmony * 0.6 + (1 - temperature) * 0.4  # Холодные цвета = доверие
        energy = saturation * 0.7 + brightness * 0.3      # Насыщенность = энергия
        calm = (1 - saturation) * 0.5 + harmony * 0.5     # Низкая насыщенность = спокойствие
        warmth = temperature * 0.8 + brightness * 0.2     # Теплые тона = теплота
        professional = harmony * 0.5 + (1 - saturation) * 0.3 + (1 - brightness) * 0.2
        
        return [trust * 10, energy * 10, calm * 10, warmth * 10, professional * 10]
    
    def _get_color_emotion(self, color: Tuple[int, int, int]) -> str:
        """Определение эмоциональной характеристики цвета."""
        r, g, b = color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        hue = h * 360
        
        if 0 <= hue <= 60 or 300 <= hue <= 360:
            return "Энергия/Страсть"
        elif 60 <= hue <= 120:
            return "Рост/Природа"
        elif 120 <= hue <= 180:
            return "Спокойствие"
        elif 180 <= hue <= 240:
            return "Доверие/Стабильность"
        elif 240 <= hue <= 300:
            return "Роскошь/Мистика"
        else:
            return "Нейтральный"
    
    def _assess_category_fit(self, color_data: Dict, category: str) -> float:
        """Оценка соответствия цветов категории."""
        # Упрощенная логика оценки
        temperature = color_data.get('color_temperature', 0.5)
        saturation = color_data.get('saturation', 0.5)
        harmony = color_data.get('harmony_score', 0.5)
        
        if category == 'Продажи':
            return min(temperature * 0.7 + saturation * 0.3, 1.0)
        elif category == 'Доверие':
            return min((1 - temperature) * 0.6 + harmony * 0.4, 1.0)
        elif category == 'Премиум':
            return min(harmony * 0.5 + (1 - saturation) * 0.5, 1.0)
        elif category == 'Молодежь':
            return min(saturation * 0.8 + temperature * 0.2, 1.0)
        elif category == 'Экология':
            return min((1 - temperature) * 0.4 + saturation * 0.3 + harmony * 0.3, 1.0)
        
        return 0.5
    
    def _translate_feature_name(self, feature: str) -> str:
        """Перевод названий признаков."""
        translations = {
            'contrast_score': 'Контрастность',
            'harmony_score': 'Цветовая гармония',
            'rule_of_thirds_score': 'Правило третей',
            'visual_balance_score': 'Визуальный баланс',
            'readability_score': 'Читаемость текста',
            'has_cta': 'Призыв к действию',
            'brightness': 'Яркость',
            'saturation': 'Насыщенность',
            'color_temperature': 'Цветовая температура',
            'text_contrast': 'Контраст текста',
            'composition_complexity': 'Сложность композиции',
            'emotional_impact': 'Эмоциональное воздействие'
        }
        return translations.get(feature, feature)
    
    def _get_factor_color(self, factor: str) -> str:
        """Цвет фактора по типу."""
        if any(word in factor.lower() for word in ['цвет', 'яркость', 'насыщенность']):
            return self.advanced_colors['emotion_color']
        elif any(word in factor.lower() for word in ['текст', 'читаемость', 'контраст']):
            return self.advanced_colors['trust_color']
        elif any(word in factor.lower() for word in ['композиция', 'баланс', 'правило']):
            return self.advanced_colors['ctr_color']
        else:
            return self.advanced_colors['performance_good']
    
    def _get_composition_color(self, value: float) -> str:
        """Цвет для значения композиции."""
        if value >= 0.8:
            return self.advanced_colors['performance_excellent']
        elif value >= 0.6:
            return self.advanced_colors['performance_good']
        elif value >= 0.4:
            return self.advanced_colors['engagement_color']
        else:
            return self.advanced_colors['performance_poor']
    
    def _translate_composition_metric(self, metric: str) -> str:
        """Перевод метрик композиции."""
        translations = {
            'rule_of_thirds_score': 'Правило третей',
            'visual_balance_score': 'Баланс',
            'symmetry_score': 'Симметрия',
            'center_focus_score': 'Центральный фокус',
            'composition_complexity': 'Сложность',
            'negative_space': 'Свободное место'
        }
        return translations.get(metric, metric)


# Алиас для обратной совместимости
Visualizer = AdvancedVisualizer
