# recommender.py
"""
Модуль рекомендаций для Creative Performance Predictor.
Генерирует персонализированные советы по улучшению креативов.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import (
    RECOMMENDATION_TYPES, RECOMMENDATION_PRIORITIES,
    SAMPLE_RECOMMENDATIONS, PERFORMANCE_METRICS
)

@dataclass
class Recommendation:
    """Класс для представления рекомендации."""
    category: str
    priority: str
    title: str
    description: str
    expected_impact: float
    confidence: float
    actionable_steps: List[str]

class RecommendationEngine:
    """
    Движок генерации рекомендаций для улучшения креативов.
    Анализирует слабые места и предлагает конкретные улучшения.
    """
    
    def __init__(self):
        self.recommendation_rules = self._initialize_rules()
        self.benchmark_values = self._get_benchmark_values()
        
    def _initialize_rules(self) -> Dict[str, Any]:
        """Инициализация правил генерации рекомендаций."""
        return {
            'color_rules': {
                'low_contrast': {
                    'threshold': 0.4,
                    'impact': 0.15,
                    'recommendation': "Увеличьте контрастность для лучшей читаемости"
                },
                'low_harmony': {
                    'threshold': 0.5,
                    'impact': 0.12,
                    'recommendation': "Улучшите цветовую гармонию для более приятного восприятия"
                },
                'extreme_temperature': {
                    'threshold': 0.2,
                    'impact': 0.10,
                    'recommendation': "Сбалансируйте цветовую температуру для целевой аудитории"
                },
                'low_saturation': {
                    'threshold': 0.3,
                    'impact': 0.08,
                    'recommendation': "Увеличьте насыщенность для большей привлекательности"
                }
            },
            'composition_rules': {
                'poor_rule_of_thirds': {
                    'threshold': 0.3,
                    'impact': 0.18,
                    'recommendation': "Расположите ключевые элементы по правилу третей"
                },
                'poor_balance': {
                    'threshold': 0.4,
                    'impact': 0.14,
                    'recommendation': "Улучшите визуальный баланс композиции"
                },
                'high_complexity': {
                    'threshold': 0.7,
                    'impact': 0.16,
                    'recommendation': "Упростите композицию для лучшего восприятия"
                },
                'poor_focus': {
                    'threshold': 0.4,
                    'impact': 0.13,
                    'recommendation': "Усильте фокусировку на главном объекте"
                }
            },
            'text_rules': {
                'poor_readability': {
                    'threshold': 0.5,
                    'impact': 0.20,
                    'recommendation': "Улучшите читаемость текстовых элементов"
                },
                'poor_hierarchy': {
                    'threshold': 0.5,
                    'impact': 0.15,
                    'recommendation': "Создайте четкую иерархию текстовых элементов"
                },
                'poor_text_contrast': {
                    'threshold': 0.4,
                    'impact': 0.18,
                    'recommendation': "Увеличьте контрастность между текстом и фоном"
                },
                'missing_cta': {
                    'threshold': 0.5,
                    'impact': 0.25,
                    'recommendation': "Добавьте четкий призыв к действию"
                },
                'excessive_text': {
                    'threshold': 0.7,
                    'impact': 0.12,
                    'recommendation': "Сократите количество текста для мобильных устройств"
                }
            }
        }
    
    def _get_benchmark_values(self) -> Dict[str, float]:
        """Получение бенчмарк значений для сравнения."""
        return {
            # Цветовые бенчмарки
            'brightness': 0.6,
            'saturation': 0.65,
            'contrast_score': 0.7,
            'harmony_score': 0.75,
            'color_temperature': 0.6,
            
            # Композиционные бенчмарки
            'rule_of_thirds_score': 0.7,
            'visual_balance_score': 0.75,
            'composition_complexity': 0.4,  # Низкая сложность лучше
            'center_focus_score': 0.6,
            
            # Текстовые бенчмарки
            'readability_score': 0.8,
            'text_hierarchy': 0.7,
            'text_contrast': 0.75,
            'text_positioning': 0.7
        }
    
    def generate_recommendations(self, 
                               image_features: Dict[str, Any],
                               predictions: Dict[str, float],
                               target_category: str = "general") -> List[Recommendation]:
        """
        Генерация персонализированных рекомендаций.
        
        Args:
            image_features: Извлеченные признаки изображения
            predictions: Предсказанные метрики эффективности
            target_category: Категория креатива для контекстуальных рекомендаций
            
        Returns:
            List[Recommendation]: Список рекомендаций, отсортированных по приоритету
        """
        recommendations = []
        
        # Анализ цветовых характеристик
        color_recs = self._analyze_color_issues(image_features)
        recommendations.extend(color_recs)
        
        # Анализ композиции
        composition_recs = self._analyze_composition_issues(image_features)
        recommendations.extend(composition_recs)
        
        # Анализ текстовых элементов
        text_recs = self._analyze_text_issues(image_features)
        recommendations.extend(text_recs)
        
        # Анализ общей эффективности
        performance_recs = self._analyze_performance_issues(predictions)
        recommendations.extend(performance_recs)
        
        # Контекстуальные рекомендации
        contextual_recs = self._get_contextual_recommendations(target_category, image_features)
        recommendations.extend(contextual_recs)
        
        # Сортировка по приоритету и ожидаемому влиянию
        recommendations.sort(key=lambda x: (
            self._get_priority_weight(x.priority),
            x.expected_impact
        ), reverse=True)
        
        # Возвращаем топ-10 рекомендаций
        return recommendations[:10]
    
    def _analyze_color_issues(self, features: Dict[str, Any]) -> List[Recommendation]:
        """Анализ проблем с цветовыми характеристиками."""
        recommendations = []
        color_rules = self.recommendation_rules['color_rules']
        
        # Проверка контрастности
        contrast = features.get('contrast_score', 0.5)
        if contrast < color_rules['low_contrast']['threshold']:
            rec = Recommendation(
                category='color',
                priority=self._determine_priority(color_rules['low_contrast']['impact']),
                title='Низкая контрастность',
                description=color_rules['low_contrast']['recommendation'],
                expected_impact=color_rules['low_contrast']['impact'],
                confidence=0.85,
                actionable_steps=[
                    "Увеличьте разность между светлыми и темными элементами",
                    "Используйте более контрастные цветовые сочетания",
                    "Проверьте читаемость на различных устройствах",
                    "Рассмотрите использование черного текста на белом фоне"
                ]
            )
            recommendations.append(rec)
        
        # Проверка цветовой гармонии
        harmony = features.get('harmony_score', 0.5)
        if harmony < color_rules['low_harmony']['threshold']:
            rec = Recommendation(
                category='color',
                priority=self._determine_priority(color_rules['low_harmony']['impact']),
                title='Нарушена цветовая гармония',
                description=color_rules['low_harmony']['recommendation'],
                expected_impact=color_rules['low_harmony']['impact'],
                confidence=0.75,
                actionable_steps=[
                    "Используйте цветовой круг для подбора гармоничных сочетаний",
                    "Примените правило 60-30-10 для распределения цветов",
                    "Ограничьтесь 3-4 основными цветами",
                    "Рассмотрите монохромную или аналогичную схему"
                ]
            )
            recommendations.append(rec)
        
        # Проверка цветовой температуры
        temperature = features.get('color_temperature', 0.5)
        if temperature < 0.2 or temperature > 0.8:
            rec = Recommendation(
                category='color',
                priority=self._determine_priority(color_rules['extreme_temperature']['impact']),
                title='Экстремальная цветовая температура',
                description=color_rules['extreme_temperature']['recommendation'],
                expected_impact=color_rules['extreme_temperature']['impact'],
                confidence=0.70,
                actionable_steps=[
                    "Добавьте теплые акценты к холодной палитре или наоборот",
                    "Учитывайте психологию восприятия цветов целевой аудиторией",
                    "Тестируйте эмоциональный отклик на цветовую схему",
                    "Балансируйте температуру в зависимости от времени показа"
                ]
            )
            recommendations.append(rec)
        
        # Проверка насыщенности
        saturation = features.get('saturation', 0.5)
        if saturation < color_rules['low_saturation']['threshold']:
            rec = Recommendation(
                category='color',
                priority=self._determine_priority(color_rules['low_saturation']['impact']),
                title='Низкая цветовая насыщенность',
                description=color_rules['low_saturation']['recommendation'],
                expected_impact=color_rules['low_saturation']['impact'],
                confidence=0.80,
                actionable_steps=[
                    "Увеличьте насыщенность ключевых элементов",
                    "Используйте яркие акцентные цвета для привлечения внимания",
                    "Сохраняйте баланс между насыщенными и приглушенными тонами",
                    "Тестируйте восприятие на различных экранах"
                ]
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _analyze_composition_issues(self, features: Dict[str, Any]) -> List[Recommendation]:
        """Анализ проблем с композицией."""
        recommendations = []
        comp_rules = self.recommendation_rules['composition_rules']
        
        # Проверка правила третей
        thirds_score = features.get('rule_of_thirds_score', 0.5)
        if thirds_score < comp_rules['poor_rule_of_thirds']['threshold']:
            rec = Recommendation(
                category='composition',
                priority=self._determine_priority(comp_rules['poor_rule_of_thirds']['impact']),
                title='Нарушено правило третей',
                description=comp_rules['poor_rule_of_thirds']['recommendation'],
                expected_impact=comp_rules['poor_rule_of_thirds']['impact'],
                confidence=0.90,
                actionable_steps=[
                    "Разместите главный объект в точках пересечения линий третей",
                    "Выровняйте горизонт по одной из горизонтальных линий",
                    "Используйте сетку третей при компоновке элементов",
                    "Избегайте центрирования всех элементов"
                ]
            )
            recommendations.append(rec)
        
        # Проверка визуального баланса
        balance = features.get('visual_balance_score', 0.5)
        if balance < comp_rules['poor_balance']['threshold']:
            rec = Recommendation(
                category='composition',
                priority=self._determine_priority(comp_rules['poor_balance']['impact']),
                title='Нарушен визуальный баланс',
                description=comp_rules['poor_balance']['recommendation'],
                expected_impact=comp_rules['poor_balance']['impact'],
                confidence=0.85,
                actionable_steps=[
                    "Перераспределите визуальный вес между частями изображения",
                    "Используйте цвет и размер для балансировки композиции",
                    "Добавьте элементы в менее загруженную область",
                    "Рассмотрите асимметричный баланс для динамичности"
                ]
            )
            recommendations.append(rec)
        
        # Проверка сложности
        complexity = features.get('composition_complexity', 0.5)
        if complexity > comp_rules['high_complexity']['threshold']:
            rec = Recommendation(
                category='composition',
                priority=self._determine_priority(comp_rules['high_complexity']['impact']),
                title='Слишком сложная композиция',
                description=comp_rules['high_complexity']['recommendation'],
                expected_impact=comp_rules['high_complexity']['impact'],
                confidence=0.80,
                actionable_steps=[
                    "Удалите второстепенные элементы",
                    "Увеличьте свободное пространство",
                    "Сгруппируйте связанные элементы",
                    "Используйте принцип 'меньше значит больше'"
                ]
            )
            recommendations.append(rec)
        
        # Проверка фокуса
        focus = features.get('center_focus_score', 0.5)
        if focus < comp_rules['poor_focus']['threshold']:
            rec = Recommendation(
                category='composition',
                priority=self._determine_priority(comp_rules['poor_focus']['impact']),
                title='Слабый центральный фокус',
                description=comp_rules['poor_focus']['recommendation'],
                expected_impact=comp_rules['poor_focus']['impact'],
                confidence=0.75,
                actionable_steps=[
                    "Выделите главный элемент размером или цветом",
                    "Используйте направляющие линии для привлечения внимания",
                    "Создайте контраст между главным объектом и фоном",
                    "Упростите фон для выделения основного содержания"
                ]
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _analyze_text_issues(self, features: Dict[str, Any]) -> List[Recommendation]:
        """Анализ проблем с текстовыми элементами."""
        recommendations = []
        text_rules = self.recommendation_rules['text_rules']
        
        # Проверка читаемости
        readability = features.get('readability_score', 0.5)
        if readability < text_rules['poor_readability']['threshold']:
            rec = Recommendation(
                category='text',
                priority=self._determine_priority(text_rules['poor_readability']['impact']),
                title='Плохая читаемость текста',
                description=text_rules['poor_readability']['recommendation'],
                expected_impact=text_rules['poor_readability']['impact'],
                confidence=0.95,
                actionable_steps=[
                    "Увеличьте размер шрифта для лучшей читаемости",
                    "Используйте простые, без засечек шрифты",
                    "Обеспечьте достаточный контраст с фоном",
                    "Избегайте размещения текста на сложном фоне"
                ]
            )
            recommendations.append(rec)
        
        # Проверка иерархии
        hierarchy = features.get('text_hierarchy', 0.5)
        if hierarchy < text_rules['poor_hierarchy']['threshold']:
            rec = Recommendation(
                category='text',
                priority=self._determine_priority(text_rules['poor_hierarchy']['impact']),
                title='Отсутствует иерархия текста',
                description=text_rules['poor_hierarchy']['recommendation'],
                expected_impact=text_rules['poor_hierarchy']['impact'],
                confidence=0.85,
                actionable_steps=[
                    "Создайте 3-4 уровня текстовой иерархии",
                    "Используйте различные размеры шрифтов",
                    "Применяйте жирность для выделения заголовков",
                    "Группируйте связанную информацию"
                ]
            )
            recommendations.append(rec)
        
        # Проверка контрастности текста
        text_contrast = features.get('text_contrast', 0.5)
        if text_contrast < text_rules['poor_text_contrast']['threshold']:
            rec = Recommendation(
                category='text',
                priority=self._determine_priority(text_rules['poor_text_contrast']['impact']),
                title='Низкий контраст текста',
                description=text_rules['poor_text_contrast']['recommendation'],
                expected_impact=text_rules['poor_text_contrast']['impact'],
                confidence=0.90,
                actionable_steps=[
                    "Используйте темный текст на светлом фоне или наоборот",
                    "Добавьте контурную обводку или тень к тексту",
                    "Создайте полупрозрачную подложку под текстом",
                    "Проверьте соответствие стандартам доступности WCAG"
                ]
            )
            recommendations.append(rec)
        
        # Проверка наличия CTA
        has_cta = features.get('has_cta', 0)
        if not has_cta:
            rec = Recommendation(
                category='text',
                priority=self._determine_priority(text_rules['missing_cta']['impact']),
                title='Отсутствует призыв к действию',
                description=text_rules['missing_cta']['recommendation'],
                expected_impact=text_rules['missing_cta']['impact'],
                confidence=0.95,
                actionable_steps=[
                    "Добавьте четкий призыв к действию (CTA)",
                    "Используйте глаголы действия: 'Купить', 'Заказать', 'Узнать больше'",
                    "Выделите CTA цветом и размером",
                    "Разместите CTA в заметном месте"
                ]
            )
            recommendations.append(rec)
        
        # Проверка количества текста
        text_amount = features.get('text_amount', 0)
        if text_amount > text_rules['excessive_text']['threshold'] * 10:  # Нормализация
            rec = Recommendation(
                category='text',
                priority=self._determine_priority(text_rules['excessive_text']['impact']),
                title='Слишком много текста',
                description=text_rules['excessive_text']['recommendation'],
                expected_impact=text_rules['excessive_text']['impact'],
                confidence=0.80,
                actionable_steps=[
                    "Сократите текст до ключевых сообщений",
                    "Используйте маркированные списки вместо абзацев",
                    "Оставьте только самую важную информацию",
                    "Рассмотрите создание отдельных версий для мобильных устройств"
                ]
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _analyze_performance_issues(self, predictions: Dict[str, float]) -> List[Recommendation]:
        """Анализ проблем с общей эффективностью."""
        recommendations = []
        
        # Анализ CTR
        ctr = predictions.get('ctr', 0)
        ctr_target = PERFORMANCE_METRICS['ctr']['target']
        
        if ctr < ctr_target * 0.7:  # Если CTR ниже 70% от цели
            rec = Recommendation(
                category='overall',
                priority='high',
                title='Низкий прогнозируемый CTR',
                description=f"Прогнозируемый CTR ({ctr*100:.2f}%) ниже целевого ({ctr_target*100:.1f}%)",
                expected_impact=0.20,
                confidence=0.85,
                actionable_steps=[
                    "Усильте визуальную привлекательность креатива",
                    "Добавьте яркие акцентные элементы",
                    "Улучшите цветовую схему для привлечения внимания",
                    "Протестируйте различные композиционные решения"
                ]
            )
            recommendations.append(rec)
        
        # Анализ конверсий
        conversion = predictions.get('conversion_rate', 0)
        conversion_target = PERFORMANCE_METRICS['conversion_rate']['target']
        
        if conversion < conversion_target * 0.7:
            rec = Recommendation(
                category='overall',
                priority='high',
                title='Низкая прогнозируемая конверсия',
                description=f"Прогнозируемая конверсия ({conversion*100:.2f}%) требует улучшения",
                expected_impact=0.25,
                confidence=0.80,
                actionable_steps=[
                    "Добавьте более убедительный призыв к действию",
                    "Четко сообщите ценностное предложение",
                    "Упростите восприятие ключевого сообщения",
                    "Используйте социальные доказательства или гарантии"
                ]
            )
            recommendations.append(rec)
        
        # Анализ вовлеченности
        engagement = predictions.get('engagement', 0)
        engagement_target = PERFORMANCE_METRICS['engagement']['target']
        
        if engagement < engagement_target * 0.7:
            rec = Recommendation(
                category='overall',
                priority='medium',
                title='Низкая прогнозируемая вовлеченность',
                description=f"Креатив может не вызвать достаточного эмоционального отклика",
                expected_impact=0.15,
                confidence=0.75,
                actionable_steps=[
                    "Добавьте эмоциональные элементы",
                    "Используйте изображения людей или лиц",
                    "Создайте интригу или любопытство",
                    "Рассмотрите добавление движения или динамики"
                ]
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _get_contextual_recommendations(self, category: str, features: Dict[str, Any]) -> List[Recommendation]:
        """Получение контекстуальных рекомендаций для конкретной категории."""
        recommendations = []
        
        contextual_rules = {
            'automotive': {
                'focus_on_emotion': {
                    'condition': lambda f: f.get('color_temperature', 0.5) < 0.4,
                    'recommendation': "Для автомобильной индустрии используйте более теплые, эмоциональные цвета"
                }
            },
            'ecommerce': {
                'clear_product_focus': {
                    'condition': lambda f: f.get('center_focus_score', 0.5) < 0.6,
                    'recommendation': "В e-commerce четко выделите продукт как главный элемент"
                }
            },
            'finance': {
                'trust_and_clarity': {
                    'condition': lambda f: f.get('composition_complexity', 0.5) > 0.6,
                    'recommendation': "Для финансовых услуг используйте простую, внушающую доверие композицию"
                }
            }
        }
        
        if category.lower() in contextual_rules:
            rules = contextual_rules[category.lower()]
            for rule_name, rule_data in rules.items():
                if rule_data['condition'](features):
                    rec = Recommendation(
                        category='contextual',
                        priority='medium',
                        title=f'Рекомендация для {category}',
                        description=rule_data['recommendation'],
                        expected_impact=0.12,
                        confidence=0.70,
                        actionable_steps=[
                            "Изучите лучшие практики вашей отрасли",
                            "Проанализируйте успешные креативы конкурентов",
                            "Учитывайте специфику целевой аудитории",
                            "Адаптируйте сообщение под отраслевые ожидания"
                        ]
                    )
                    recommendations.append(rec)
        
        return recommendations
    
    def _determine_priority(self, impact: float) -> str:
        """Определение приоритета рекомендации на основе ожидаемого влияния."""
        if impact >= RECOMMENDATION_PRIORITIES['high']['min_impact']:
            return 'high'
        elif impact >= RECOMMENDATION_PRIORITIES['medium']['min_impact']:
            return 'medium'
        else:
            return 'low'
    
    def _get_priority_weight(self, priority: str) -> int:
        """Получение веса приоритета для сортировки."""
        weights = {'high': 3, 'medium': 2, 'low': 1}
        return weights.get(priority, 1)
    
    def generate_optimization_suggestions(self, 
                                        current_features: Dict[str, Any],
                                        target_improvement: float = 0.2) -> Dict[str, Any]:
        """
        Генерация конкретных предложений по оптимизации.
        
        Args:
            current_features: Текущие характеристики креатива
            target_improvement: Целевое улучшение (в долях)
            
        Returns:
            Dict[str, Any]: Предложения по оптимизации
        """
        optimizations = {
            'color_optimizations': [],
            'composition_optimizations': [],
            'text_optimizations': [],
            'priority_actions': []
        }
        
        # Цветовые оптимизации
        if current_features.get('contrast_score', 0) < self.benchmark_values['contrast_score']:
            target_contrast = min(
                current_features.get('contrast_score', 0) + target_improvement,
                1.0
            )
            optimizations['color_optimizations'].append({
                'parameter': 'contrast_score',
                'current_value': current_features.get('contrast_score', 0),
                'target_value': target_contrast,
                'improvement': target_contrast - current_features.get('contrast_score', 0),
                'action': 'Увеличить контрастность между элементами'
            })
        
        # Композиционные оптимизации
        if current_features.get('rule_of_thirds_score', 0) < self.benchmark_values['rule_of_thirds_score']:
            target_thirds = min(
                current_features.get('rule_of_thirds_score', 0) + target_improvement,
                1.0
            )
            optimizations['composition_optimizations'].append({
                'parameter': 'rule_of_thirds_score',
                'current_value': current_features.get('rule_of_thirds_score', 0),
                'target_value': target_thirds,
                'improvement': target_thirds - current_features.get('rule_of_thirds_score', 0),
                'action': 'Переместить ключевые элементы в точки силы'
            })
        
        # Текстовые оптимизации
        if current_features.get('readability_score', 0) < self.benchmark_values['readability_score']:
            target_readability = min(
                current_features.get('readability_score', 0) + target_improvement,
                1.0
            )
            optimizations['text_optimizations'].append({
                'parameter': 'readability_score',
                'current_value': current_features.get('readability_score', 0),
                'target_value': target_readability,
                'improvement': target_readability - current_features.get('readability_score', 0),
                'action': 'Улучшить читаемость текста'
            })
        
        # Приоритетные действия (топ-3 по влиянию)
        all_optimizations = (
            optimizations['color_optimizations'] +
            optimizations['composition_optimizations'] +
            optimizations['text_optimizations']
        )
        
        # Сортировка по потенциальному улучшению
        all_optimizations.sort(key=lambda x: x['improvement'], reverse=True)
        optimizations['priority_actions'] = all_optimizations[:3]
        
        return optimizations
    
    def create_action_plan(self, recommendations: List[Recommendation]) -> Dict[str, Any]:
        """
        Создание пошагового плана действий.
        
        Args:
            recommendations: Список рекомендаций
            
        Returns:
            Dict[str, Any]: Структурированный план действий
        """
        # Группировка по приоритетам
        high_priority = [r for r in recommendations if r.priority == 'high']
        medium_priority = [r for r in recommendations if r.priority == 'medium']
        low_priority = [r for r in recommendations if r.priority == 'low']
        
        # Расчет общего потенциального улучшения
        total_impact = sum(r.expected_impact for r in recommendations)
        
        action_plan = {
            'immediate_actions': [
                {
                    'title': rec.title,
                    'description': rec.description,
                    'steps': rec.actionable_steps[:2],  # Первые 2 шага
                    'expected_impact': rec.expected_impact,
                    'estimated_time': '1-2 часа'
                }
                for rec in high_priority[:3]
            ],
            
            'short_term_actions': [
                {
                    'title': rec.title,
                    'description': rec.description,
                    'steps': rec.actionable_steps,
                    'expected_impact': rec.expected_impact,
                    'estimated_time': '2-4 часа'
                }
                for rec in medium_priority[:4]
            ],
            
            'long_term_improvements': [
                {
                    'title': rec.title,
                    'description': rec.description,
                    'steps': rec.actionable_steps,
                    'expected_impact': rec.expected_impact,
                    'estimated_time': '4+ часов'
                }
                for rec in low_priority
            ],
            
            'summary': {
                'total_recommendations': len(recommendations),
                'high_priority_count': len(high_priority),
                'potential_improvement': f"{total_impact:.1%}",
                'estimated_total_time': '8-12 часов',
                'key_focus_areas': list(set(r.category for r in recommendations[:5]))
            }
        }
        
        return action_plan