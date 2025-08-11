# ml_engine.py
"""
Модуль машинного обучения для Creative Performance Predictor.
Обучает модели предсказания эффективности креативов.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from config import (
    ML_MODELS, PERFORMANCE_METRICS, SYNTHETIC_DATA, 
    CREATIVE_CATEGORIES, REGIONS, FEATURE_IMPORTANCE_THRESHOLD
)

class MLEngine:
    """
    Движок машинного обучения для предсказания эффективности креативов.
    Включает обучение моделей, предсказания и объяснения результатов.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        self.training_data = None
        self.feature_importance = {}
        
        # Инициализация моделей
        self._initialize_models()
        
    def _initialize_models(self):
        """Инициализация ML моделей."""
        # Random Forest
        self.models['random_forest'] = RandomForestRegressor(
            **ML_MODELS['random_forest']
        )
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBRegressor(
            **ML_MODELS['xgboost'],
            objective='reg:squarederror'
        )
        
        # Linear Ridge
        self.models['ridge'] = Ridge(
            **ML_MODELS['linear']
        )
        
        # Скалеры для каждой модели
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def generate_synthetic_data(self, n_samples: int = None) -> pd.DataFrame:
        """
        Генерация синтетических данных для обучения модели.
        
        Args:
            n_samples: Количество образцов для генерации
            
        Returns:
            pd.DataFrame: Синтетический датасет
        """
        if n_samples is None:
            n_samples = SYNTHETIC_DATA['n_samples']
            
        np.random.seed(SYNTHETIC_DATA['random_state'])
        
        # Генерация признаков изображений
        data = {
            # Цветовые признаки
            'brightness': np.random.beta(2, 2, n_samples),
            'saturation': np.random.beta(2, 3, n_samples),
            'contrast_score': np.random.beta(3, 2, n_samples),
            'color_temperature': np.random.beta(2, 2, n_samples),
            'harmony_score': np.random.beta(3, 2, n_samples),
            'color_diversity': np.random.poisson(3, n_samples) + 1,
            'warm_cool_ratio': np.random.beta(2, 2, n_samples),
            
            # Композиционные признаки
            'rule_of_thirds_score': np.random.beta(2, 3, n_samples),
            'visual_balance_score': np.random.beta(3, 2, n_samples),
            'composition_complexity': np.random.beta(2, 3, n_samples),
            'center_focus_score': np.random.beta(2, 2, n_samples),
            'leading_lines_score': np.random.beta(1, 4, n_samples),
            'symmetry_score': np.random.beta(2, 3, n_samples),
            'depth_perception': np.random.beta(2, 2, n_samples),
            
            # Текстовые признаки
            'text_amount': np.random.poisson(3, n_samples),
            'total_characters': np.random.poisson(50, n_samples),
            'readability_score': np.random.beta(3, 2, n_samples),
            'text_hierarchy': np.random.beta(3, 2, n_samples),
            'text_positioning': np.random.beta(2, 2, n_samples),
            'text_contrast': np.random.beta(3, 2, n_samples),
            'has_cta': np.random.binomial(1, 0.7, n_samples),
            
            # Дополнительные признаки
            'aspect_ratio': np.random.lognormal(0, 0.3, n_samples),
            'image_size_score': np.random.beta(2, 2, n_samples),
            
            # Контекстуальные переменные
            'category': np.random.choice(CREATIVE_CATEGORIES, n_samples),
            'region': np.random.choice(REGIONS, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Нормализация некоторых признаков
        df['color_diversity'] = np.clip(df['color_diversity'], 1, 10) / 10.0
        df['text_amount'] = np.clip(df['text_amount'], 0, 10) / 10.0
        df['total_characters'] = np.clip(df['total_characters'], 0, 200) / 200.0
        df['aspect_ratio'] = np.clip(df['aspect_ratio'], 0.1, 5.0) / 5.0
        
        # Генерация целевых метрик на основе признаков
        df = self._generate_target_metrics(df)
        
        return df
    
    def _generate_target_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация целевых метрик эффективности на основе признаков.
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            pd.DataFrame: DataFrame с добавленными целевыми метриками
        """
        n_samples = len(df)
        noise_level = SYNTHETIC_DATA['noise_level']
        
        # CTR модель (основана на визуальной привлекательности)
        ctr_base = (
            0.4 * df['brightness'] +
            0.3 * df['contrast_score'] +
            0.2 * df['color_temperature'] +
            0.1 * df['harmony_score'] +
            0.2 * df['rule_of_thirds_score'] +
            0.1 * df['visual_balance_score'] +
            -0.2 * df['composition_complexity'] +  # Сложность снижает CTR
            0.1 * df['has_cta']
        )
        
        # Добавление шума и нормализация
        ctr_noise = np.random.normal(0, noise_level, n_samples)
        df['ctr'] = np.clip(ctr_base + ctr_noise, 0.001, 0.1)
        
        # Conversion Rate модель (основана на убедительности)
        conv_base = (
            0.3 * df['readability_score'] +
            0.3 * df['text_contrast'] +
            0.2 * df['text_hierarchy'] +
            0.4 * df['has_cta'] +
            0.2 * df['visual_balance_score'] +
            0.1 * df['center_focus_score'] +
            -0.1 * df['composition_complexity']
        )
        
        conv_noise = np.random.normal(0, noise_level, n_samples)
        df['conversion_rate'] = np.clip(conv_base + conv_noise, 0.001, 0.5)
        
        # Engagement модель (основана на эмоциональной привлекательности)
        eng_base = (
            0.3 * df['color_temperature'] +
            0.2 * df['saturation'] +
            0.2 * df['harmony_score'] +
            0.1 * df['depth_perception'] +
            0.1 * df['leading_lines_score'] +
            0.1 * df['symmetry_score'] +
            -0.1 * df['composition_complexity']
        )
        
        eng_noise = np.random.normal(0, noise_level, n_samples)
        df['engagement'] = np.clip(eng_base + eng_noise, 0.01, 1.0)
        
        # Категориальные корректировки
        category_multipliers = {
            'Автомобили': {'ctr': 1.2, 'conversion_rate': 0.9, 'engagement': 1.1},
            'E-commerce': {'ctr': 1.1, 'conversion_rate': 1.3, 'engagement': 1.0},
            'Финансы': {'ctr': 0.8, 'conversion_rate': 1.1, 'engagement': 0.9},
            'Технологии': {'ctr': 1.0, 'conversion_rate': 1.2, 'engagement': 1.1}
        }
        
        for category, multipliers in category_multipliers.items():
            mask = df['category'] == category
            for metric, mult in multipliers.items():
                df.loc[mask, metric] *= mult
        
        # Региональные корректировки
        region_multipliers = {
            'Россия': {'ctr': 1.1, 'conversion_rate': 1.0, 'engagement': 1.2},
            'США': {'ctr': 1.0, 'conversion_rate': 1.1, 'engagement': 1.0},
            'Европа': {'ctr': 0.9, 'conversion_rate': 1.0, 'engagement': 0.9}
        }
        
        for region, multipliers in region_multipliers.items():
            mask = df['region'] == region
            for metric, mult in multipliers.items():
                df.loc[mask, metric] *= mult
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Подготовка признаков для обучения модели.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            Tuple[np.ndarray, List[str]]: Массив признаков и список их названий
        """
        # Исключаем целевые переменные и категориальные признаки
        feature_columns = [col for col in df.columns 
                          if col not in ['ctr', 'conversion_rate', 'engagement', 'category', 'region']]
        
        # One-hot encoding для категориальных переменных
        df_encoded = pd.get_dummies(df, columns=['category', 'region'], prefix=['cat', 'reg'])
        
        # Обновляем список признаков
        feature_columns = [col for col in df_encoded.columns 
                          if col not in ['ctr', 'conversion_rate', 'engagement']]
        
        X = df_encoded[feature_columns].values
        
        return X, feature_columns
    
    def train_models(self, df: pd.DataFrame = None) -> Dict[str, float]:
        """
        Обучение всех моделей на данных.
        
        Args:
            df: DataFrame с обучающими данными
            
        Returns:
            Dict[str, float]: Метрики качества моделей
        """
        if df is None:
            df = self.generate_synthetic_data()
        
        self.training_data = df
        
        # Подготовка признаков
        X, feature_names = self.prepare_features(df)
        self.feature_names = feature_names
        
        # Целевые переменные
        targets = ['ctr', 'conversion_rate', 'engagement']
        
        results = {}
        
        for target in targets:
            y = df[target].values
            
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            target_results = {}
            
            for model_name, model in self.models.items():
                # Масштабирование признаков
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
                
                # Обучение модели
                model.fit(X_train_scaled, y_train)
                
                # Предсказания
                y_pred = model.predict(X_test_scaled)
                
                # Метрики качества
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                target_results[model_name] = {
                    'r2_score': r2,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Сохранение важности признаков (для Random Forest и XGBoost)
                if hasattr(model, 'feature_importances_'):
                    importance_key = f"{target}_{model_name}"
                    self.feature_importance[importance_key] = dict(
                        zip(feature_names, model.feature_importances_)
                    )
            
            results[target] = target_results
        
        self.is_trained = True
        return results
    
    def predict(self, features: Dict[str, Any], model_name: str = 'random_forest') -> Dict[str, float]:
        """
        Предсказание эффективности креатива.
        
        Args:
            features: Словарь с признаками изображения
            model_name: Название модели для предсказания
            
        Returns:
            Dict[str, float]: Предсказанные метрики эффективности
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Вызовите train_models() сначала.")
        
        # Подготовка признаков для предсказания
        feature_vector = self._prepare_single_prediction(features)
        
        # Масштабирование
        feature_vector_scaled = self.scalers[model_name].transform([feature_vector])
        
        # Предсказания для всех целевых переменных
        predictions = {}
        targets = ['ctr', 'conversion_rate', 'engagement']
        
        for target in targets:
            pred = self.models[model_name].predict(feature_vector_scaled)[0]
            
            # Ограничение предсказаний в разумных пределах
            min_val = PERFORMANCE_METRICS[target]['min']
            max_val = PERFORMANCE_METRICS[target]['max']
            predictions[target] = np.clip(pred, min_val, max_val)
        
        return predictions
    
    def get_prediction_confidence(self, features: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """
        Получить доверительные интервалы для предсказаний.
        
        Args:
            features: Словарь с признаками изображения
            
        Returns:
            Dict[str, Tuple[float, float]]: Доверительные интервалы (нижняя, верхняя границы)
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена.")
        
        # Используем Random Forest для оценки неопределенности
        feature_vector = self._prepare_single_prediction(features)
        feature_vector_scaled = self.scalers['random_forest'].transform([feature_vector])
        
        confidence_intervals = {}
        targets = ['ctr', 'conversion_rate', 'engagement']
        
        for target in targets:
            # Предсказания всех деревьев в лесу
            tree_predictions = []
            for tree in self.models['random_forest'].estimators_:
                pred = tree.predict(feature_vector_scaled)[0]
                tree_predictions.append(pred)
            
            # Расчет доверительного интервала (95%)
            predictions_array = np.array(tree_predictions)
            lower_bound = np.percentile(predictions_array, 2.5)
            upper_bound = np.percentile(predictions_array, 97.5)
            
            # Ограничение в разумных пределах
            min_val = PERFORMANCE_METRICS[target]['min']
            max_val = PERFORMANCE_METRICS[target]['max']
            
            confidence_intervals[target] = (
                np.clip(lower_bound, min_val, max_val),
                np.clip(upper_bound, min_val, max_val)
            )
        
        return confidence_intervals
    
    def get_feature_importance(self, target: str = 'ctr', model_name: str = 'random_forest') -> List[Tuple[str, float]]:
        """
        Получить важность признаков для конкретной модели и цели.
        
        Args:
            target: Целевая переменная
            model_name: Название модели
            
        Returns:
            List[Tuple[str, float]]: Список кортежей (признак, важность)
        """
        importance_key = f"{target}_{model_name}"
        
        if importance_key not in self.feature_importance:
            return []
        
        # Фильтрация и сортировка по важности
        importance_dict = self.feature_importance[importance_key]
        filtered_importance = [
            (feature, importance) for feature, importance in importance_dict.items()
            if importance >= FEATURE_IMPORTANCE_THRESHOLD
        ]
        
        # Сортировка по убыванию важности
        filtered_importance.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_importance[:15]  # Топ-15 признаков
    
    def explain_prediction(self, features: Dict[str, Any], target: str = 'ctr') -> Dict[str, Any]:
        """
        Объяснение предсказания модели.
        
        Args:
            features: Словарь с признаками изображения
            target: Целевая переменная для объяснения
            
        Returns:
            Dict[str, Any]: Объяснение предсказания
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена.")
        
        # Предсказание
        predictions = self.predict(features)
        confidence = self.get_prediction_confidence(features)
        
        # Важность признаков
        feature_importance = self.get_feature_importance(target)
        
        # Анализ влияния конкретных значений признаков
        feature_vector = self._prepare_single_prediction(features)
        feature_impacts = []
        
        for feature_name, importance in feature_importance[:10]:  # Топ-10
            feature_idx = self.feature_names.index(feature_name)
            feature_value = feature_vector[feature_idx]
            
            # Простое правило для определения влияния
            if importance > 0.05:  # Значимый признак
                if feature_value > 0.7:
                    impact = "положительное"
                elif feature_value < 0.3:
                    impact = "отрицательное" 
                else:
                    impact = "нейтральное"
            else:
                impact = "минимальное"
            
            feature_impacts.append({
                'feature': feature_name,
                'value': feature_value,
                'importance': importance,
                'impact': impact
            })
        
        # Общая оценка качества креатива
        overall_score = (
            predictions['ctr'] / PERFORMANCE_METRICS['ctr']['target'] * 0.4 +
            predictions['conversion_rate'] / PERFORMANCE_METRICS['conversion_rate']['target'] * 0.4 +
            predictions['engagement'] / PERFORMANCE_METRICS['engagement']['target'] * 0.2
        )
        
        explanation = {
            'predictions': predictions,
            'confidence_intervals': confidence,
            'overall_score': min(overall_score, 1.0),
            'feature_impacts': feature_impacts,
            'performance_category': self._categorize_performance(overall_score),
            'key_insights': self._generate_insights(feature_impacts, predictions)
        }
        
        return explanation
    
    def _prepare_single_prediction(self, features: Dict[str, Any]) -> np.ndarray:
        """Подготовка одного образца для предсказания."""
        # Заполнение значений по умолчанию для отсутствующих признаков
        default_features = {name: 0.5 for name in self.feature_names}
        default_features.update(features)
        
        # Создание вектора признаков в правильном порядке
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name.startswith('cat_') or feature_name.startswith('reg_'):
                # Категориальные признаки - бинарные
                feature_vector.append(0.0)
            else:
                feature_vector.append(default_features.get(feature_name, 0.5))
        
        return np.array(feature_vector)
    
    def _categorize_performance(self, score: float) -> str:
        """Категоризация общей эффективности."""
        if score >= 0.8:
            return "Отличная"
        elif score >= 0.6:
            return "Хорошая"
        elif score >= 0.4:
            return "Средняя"
        else:
            return "Требует улучшения"
    
    def _generate_insights(self, feature_impacts: List[Dict], predictions: Dict[str, float]) -> List[str]:
        """Генерация ключевых инсайтов по анализу."""
        insights = []
        
        # Анализ цветовых характеристик
        color_features = [f for f in feature_impacts if any(color_term in f['feature'] 
                         for color_term in ['brightness', 'saturation', 'contrast', 'color'])]
        
        if color_features:
            good_color_features = [f for f in color_features if f['impact'] == 'положительное']
            if len(good_color_features) >= 2:
                insights.append("Цветовая палитра креатива способствует привлечению внимания")
            else:
                insights.append("Рекомендуется оптимизировать цветовые решения")
        
        # Анализ композиции
        composition_features = [f for f in feature_impacts if any(comp_term in f['feature'] 
                               for comp_term in ['rule_of_thirds', 'balance', 'composition'])]
        
        if composition_features:
            good_comp_features = [f for f in composition_features if f['impact'] == 'положительное']
            if good_comp_features:
                insights.append("Композиция креатива следует дизайнерским принципам")
            else:
                insights.append("Композиция требует доработки для улучшения восприятия")
        
        # Анализ текстовых элементов
        text_features = [f for f in feature_impacts if 'text' in f['feature']]
        
        if text_features:
            good_text_features = [f for f in text_features if f['impact'] == 'положительное']
            if len(good_text_features) >= 2:
                insights.append("Текстовые элементы хорошо читаемы и структурированы")
            else:
                insights.append("Текстовые элементы нуждаются в оптимизации")
        
        # Общие рекомендации на основе предсказаний
        if predictions['ctr'] < PERFORMANCE_METRICS['ctr']['target']:
            insights.append("Креатив может не привлекать достаточного внимания аудитории")
        
        if predictions['conversion_rate'] < PERFORMANCE_METRICS['conversion_rate']['target']:
            insights.append("Креатив может слабо мотивировать к совершению действий")
        
        return insights[:5]  # Максимум 5 инсайтов
    
    def save_model(self, filepath: str):
        """Сохранение обученной модели."""
        if not self.is_trained:
            raise ValueError("Нет обученной модели для сохранения.")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Загрузка сохраненной модели."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.is_trained = model_data['is_trained']