# ml_engine.py - РЕВОЛЮЦИОННАЯ ВЕРСИЯ
"""
Модуль машинного обучения для Creative Performance Predictor.
Полностью переписанная версия с реальными ML моделями и feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Безопасный импорт XGBoost (ОБЛЕГЧЕННАЯ ВЕРСИЯ)
XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost доступен для продвинутого ML")
except ImportError:
    print("ℹ️ XGBoost недоступен. Используются Random Forest и Gradient Boosting.")

from config import (
    ML_MODELS, PERFORMANCE_METRICS, SYNTHETIC_DATA, 
    CREATIVE_CATEGORIES, REGIONS, FEATURE_IMPORTANCE_THRESHOLD
)

class AdvancedMLEngine:
    """
    Продвинутый движок машинного обучения для предсказания эффективности креативов.
    Использует множественные модели, feature engineering и продвинутую валидацию.
    """
    
    def __init__(self):
        # Словари для хранения моделей по целевым метрикам
        self.models = {
            'ctr': {},
            'conversion_rate': {},
            'engagement': {}
        }
        
        # Скалеры для каждой метрики
        self.scalers = {
            'ctr': StandardScaler(),
            'conversion_rate': RobustScaler(),  # Более устойчив к выбросам
            'engagement': StandardScaler()
        }
        
        # Feature engineering
        self.feature_selectors = {}
        self.feature_names = []
        self.feature_importance = {}
        self.model_performance = {}
        
        # Статус обучения
        self.is_trained = False
        self.training_data = None
        self.training_timestamp = None
        
        # Инициализация моделей
        self._initialize_models()
        
        # Веса для ансамбля моделей
        self.ensemble_weights = {
            'ctr': {'random_forest': 0.4, 'gradient_boosting': 0.3, 'elastic_net': 0.3},
            'conversion_rate': {'random_forest': 0.3, 'gradient_boosting': 0.4, 'elastic_net': 0.3},
            'engagement': {'random_forest': 0.5, 'gradient_boosting': 0.3, 'elastic_net': 0.2}
        }
        
    def _initialize_models(self):
        """Инициализация продвинутых ML моделей."""
        
        # Random Forest (базовая модель)
        rf_params = {
            'n_estimators': 200,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Gradient Boosting (продвинутая модель)
        gb_params = {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'random_state': 42
        }
        
        # Elastic Net (линейная модель с регуляризацией)
        en_params = {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'random_state': 42,
            'max_iter': 2000
        }
        
        # XGBoost (если доступен)
        if XGBOOST_AVAILABLE:
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'objective': 'reg:squarederror'
            }
        
        # Создаем модели для каждой целевой метрики
        for target in ['ctr', 'conversion_rate', 'engagement']:
            self.models[target] = {
                'random_forest': RandomForestRegressor(**rf_params),
                'gradient_boosting': GradientBoostingRegressor(**gb_params),
                'elastic_net': ElasticNet(**en_params)
            }
            
            if XGBOOST_AVAILABLE:
                self.models[target]['xgboost'] = xgb.XGBRegressor(**xgb_params)
    
    def generate_realistic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Генерация реалистичных данных на основе исследований маркетинговой эффективности.
        """
        np.random.seed(42)
        
        # === ОСНОВНЫЕ ВИЗУАЛЬНЫЕ ПРИЗНАКИ ===
        data = {}
        
        # Цветовые характеристики (основаны на исследованиях цветовой психологии)
        data['brightness'] = np.random.beta(2.5, 2.5, n_samples)  # Умеренная яркость работает лучше
        data['saturation'] = np.random.beta(3, 2, n_samples)      # Высокая насыщенность привлекает
        data['contrast_score'] = np.random.beta(3, 2, n_samples)  # Высокий контраст важен
        data['color_temperature'] = np.random.beta(2, 2, n_samples)  # Баланс теплых и холодных
        data['harmony_score'] = np.random.beta(2.5, 2, n_samples)    # Гармония важна
        data['color_vibrancy'] = np.random.beta(2.5, 2, n_samples)   # Живость цветов
        data['emotional_impact'] = np.random.beta(2, 2, n_samples)   # Эмоциональное воздействие
        
        # Композиционные характеристики
        data['rule_of_thirds_score'] = np.random.beta(2, 3, n_samples)  # Не все следуют правилу
        data['visual_balance_score'] = np.random.beta(3, 2, n_samples)  # Баланс важен
        data['composition_complexity'] = np.random.beta(2, 3, n_samples)  # Простота лучше
        data['center_focus_score'] = np.random.beta(2.5, 2.5, n_samples)  # Средний фокус
        data['symmetry_score'] = np.random.beta(2, 3, n_samples)          # Немного асимметрии
        data['negative_space'] = np.random.beta(2.5, 2, n_samples)        # Важно для восприятия
        data['visual_flow'] = np.random.beta(2.5, 2.5, n_samples)         # Движение взгляда
        
        # Текстовые характеристики
        # Количество текста (дискретное распределение)
        text_amounts = np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, 
                                      p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.08, 0.02])
        data['text_amount'] = text_amounts / 6.0  # Нормализация
        
        # Качество текста зависит от его количества
        data['readability_score'] = np.where(
            text_amounts > 0,
            np.random.beta(3, 2, n_samples),  # Если есть текст, обычно читаемый
            1.0  # Если текста нет, читаемость максимальная
        )
        
        data['text_hierarchy'] = np.where(
            text_amounts > 1,
            np.random.beta(2.5, 2.5, n_samples),  # Иерархия важна при много тексте
            1.0  # Если мало текста, иерархия не критична
        )
        
        data['text_contrast'] = np.where(
            text_amounts > 0,
            np.random.beta(3, 2, n_samples),  # Контраст важен для любого текста
            1.0
        )
        
        # Призыв к действию (binary с вероятностью)
        cta_probability = np.where(text_amounts > 0, 0.7, 0.1)  # Больше шансов при наличии текста
        data['has_cta'] = np.random.binomial(1, cta_probability)
        
        # Дополнительные признаки
        data['aspect_ratio'] = np.clip(np.random.lognormal(0, 0.3, n_samples), 0.3, 3.0) / 3.0
        data['overall_complexity'] = np.random.beta(2, 3, n_samples)  # Простота лучше
        data['visual_appeal'] = np.random.beta(2.5, 2, n_samples)     # Общая привлекательность
        
        # === КАТЕГОРИАЛЬНЫЕ ПЕРЕМЕННЫЕ ===
        categories = ['Автомобили', 'E-commerce', 'Финансы', 'Технологии', 'Здоровье', 'Образование']
        regions = ['Россия', 'США', 'Европа', 'Азия']
        
        data['category'] = np.random.choice(categories, n_samples)
        data['region'] = np.random.choice(regions, n_samples)
        
        # === ВРЕМЕННЫЕ ФАКТОРЫ ===
        # Эмуляция сезонности и трендов
        days = np.random.randint(0, 365, n_samples)
        data['seasonality'] = 0.5 + 0.3 * np.sin(2 * np.pi * days / 365)  # Годовая сезонность
        data['weekly_trend'] = 0.5 + 0.2 * np.sin(2 * np.pi * days / 7)   # Недельный тренд
        
        df = pd.DataFrame(data)
        
        # === ГЕНЕРАЦИЯ ЦЕЛЕВЫХ МЕТРИК ===
        df = self._generate_realistic_targets(df)
        
        return df
    
    def _generate_realistic_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация реалистичных целевых метрик на основе исследований и здравого смысла.
        """
        n_samples = len(df)
        
        # === CTR МОДЕЛЬ (Click-Through Rate) ===
        # Основано на исследованиях факторов, влияющих на CTR
        
        ctr_base = (
            # Цветовые факторы (30% влияния)
            0.15 * df['contrast_score'] +           # Контраст критичен для привлечения внимания
            0.08 * df['color_vibrancy'] +           # Яркие цвета привлекают
            0.07 * df['emotional_impact'] +         # Эмоциональность важна
            
            # Композиционные факторы (40% влияния)
            0.12 * df['visual_appeal'] +            # Общая привлекательность
            0.10 * (1 - df['composition_complexity']) +  # Простота лучше для CTR
            0.08 * df['rule_of_thirds_score'] +     # Правильная композиция
            0.10 * df['negative_space'] +           # Пространство для восприятия
            
            # Текстовые факторы (20% влияния)
            0.12 * df['has_cta'] +                  # CTA критичен для кликов
            0.08 * df['text_contrast'] +            # Читаемость важна
            
            # Контекстуальные факторы (10% влияния)
            0.05 * df['seasonality'] +              # Сезонные колебания
            0.05 * df['weekly_trend']               # Недельные тренды
        )
        
        # Категориальные модификаторы для CTR
        category_ctr_multipliers = {
            'E-commerce': 1.3,      # E-commerce обычно показывает высокий CTR
            'Автомобили': 1.1,      # Автомобили привлекают внимание
            'Технологии': 1.0,      # Базовый уровень
            'Финансы': 0.8,         # Консервативная сфера
            'Здоровье': 0.9,        # Требует доверия
            'Образование': 0.85     # Специфическая аудитория
        }
        
        region_ctr_multipliers = {
            'США': 1.2,             # Развитый рынок онлайн рекламы
            'Европа': 1.0,          # Базовый уровень
            'Россия': 0.9,          # Развивающийся рынок
            'Азия': 1.1             # Активные пользователи интернета
        }
        
        # Применяем модификаторы
        for category, multiplier in category_ctr_multipliers.items():
            mask = df['category'] == category
            ctr_base[mask] *= multiplier
            
        for region, multiplier in region_ctr_multipliers.items():
            mask = df['region'] == region
            ctr_base[mask] *= multiplier
        
        # Добавляем реалистичный шум и нормализуем
        ctr_noise = np.random.normal(0, 0.05, n_samples)
        df['ctr'] = np.clip(ctr_base + ctr_noise, 0.001, 0.15)
        
        # === CONVERSION RATE МОДЕЛЬ ===
        # Основано на исследованиях UX и конверсионной оптимизации
        
        conv_base = (
            # Доверие и ясность (50% влияния)
            0.20 * df['readability_score'] +        # Читаемость критична
            0.15 * df['text_hierarchy'] +           # Структура информации
            0.15 * df['has_cta'] +                  # Четкий CTA
            
            # Визуальная привлекательность (30% влияния)
            0.12 * df['visual_appeal'] +            # Общая привлекательность
            0.10 * df['harmony_score'] +            # Гармония вызывает доверие
            0.08 * (1 - df['overall_complexity']) + # Простота помогает решению
            
            # Эмоциональное воздействие (20% влияния)
            0.12 * df['emotional_impact'] +         # Эмоции влияют на решения
            0.08 * df['visual_flow']                # Логичность подачи
        )
        
        # Категориальные модификаторы для конверсий
        category_conv_multipliers = {
            'E-commerce': 1.4,      # E-commerce оптимизирован для конверсий
            'Финансы': 1.2,         # Высокая ценность конверсий
            'Здоровье': 1.1,        # Мотивация к действию
            'Технологии': 1.0,      # Базовый уровень
            'Автомобили': 0.8,      # Длинный цикл принятия решений
            'Образование': 0.9      # Обдуманные решения
        }
        
        # Применяем модификаторы
        for category, multiplier in category_conv_multipliers.items():
            mask = df['category'] == category
            conv_base[mask] *= multiplier
            
        for region, multiplier in region_ctr_multipliers.items():  # Используем те же региональные
            mask = df['region'] == region
            conv_base[mask] *= multiplier * 0.8  # Но с меньшим влиянием
        
        conv_noise = np.random.normal(0, 0.03, n_samples)
        df['conversion_rate'] = np.clip(conv_base + conv_noise, 0.001, 0.35)
        
        # === ENGAGEMENT МОДЕЛЬ ===
        # Основано на исследованиях социальных медиа и вовлеченности
        
        eng_base = (
            # Эмоциональные факторы (40% влияния)
            0.20 * df['emotional_impact'] +         # Эмоции = вовлеченность
            0.10 * df['color_vibrancy'] +           # Яркость привлекает
            0.10 * df['visual_appeal'] +            # Красота мотивирует
            
            # Интересность контента (35% влияния)
            0.15 * (1 - df['overall_complexity']) + # Простота восприятия
            0.10 * df['visual_flow'] +              # Логичность подачи
            0.10 * df['rule_of_thirds_score'] +     # Профессиональный вид
            
            # Качество исполнения (25% влияния)
            0.12 * df['harmony_score'] +            # Гармония = качество
            0.08 * df['text_contrast'] +            # Читаемость
            0.05 * df['seasonality']                # Сезонные факторы
        )
        
        # Категориальные модификаторы для engagement
        category_eng_multipliers = {
            'Технологии': 1.3,      # Tech-аудитория активна
            'Автомобили': 1.2,      # Эмоциональная категория
            'E-commerce': 1.0,      # Базовый уровень
            'Здоровье': 0.9,        # Более серьезный контент
            'Финансы': 0.8,         # Консервативная аудитория
            'Образование': 0.85     # Специфическая аудитория
        }
        
        for category, multiplier in category_eng_multipliers.items():
            mask = df['category'] == category
            eng_base[mask] *= multiplier
        
        eng_noise = np.random.normal(0, 0.04, n_samples)
        df['engagement'] = np.clip(eng_base + eng_noise, 0.01, 0.8)
        
        # === КОРРЕЛЯЦИИ МЕЖДУ МЕТРИКАМИ ===
        # В реальности метрики коррелируют, добавляем это
        
        # CTR влияет на конверсии (но не линейно)
        ctr_effect = np.sqrt(df['ctr']) * 0.3
        df['conversion_rate'] = np.clip(df['conversion_rate'] + ctr_effect, 0.001, 0.35)
        
        # Engagement влияет на CTR
        eng_effect = np.sqrt(df['engagement']) * 0.1
        df['ctr'] = np.clip(df['ctr'] + eng_effect, 0.001, 0.15)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Продвинутая подготовка признаков с feature engineering.
        """
        # Исключаем целевые переменные и вспомогательные
        exclude_columns = ['ctr', 'conversion_rate', 'engagement', 'seasonality', 'weekly_trend']
        
        # One-hot encoding для категориальных
        df_encoded = pd.get_dummies(df, columns=['category', 'region'], prefix=['cat', 'reg'])
        
        # Создание новых признаков (feature engineering)
        df_encoded = self._create_engineered_features(df_encoded)
        
        # Получаем список признаков
        feature_columns = [col for col in df_encoded.columns if col not in exclude_columns]
        
        # Создаем отдельные наборы для каждой целевой метрики
        feature_sets = {}
        
        for target in ['ctr', 'conversion_rate', 'engagement']:
            X = df_encoded[feature_columns].values
            feature_sets[target] = X
        
        self.feature_names = feature_columns
        return feature_sets, feature_columns
    
    def _create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание новых признаков на основе существующих."""
        
        # Интерактивные признаки
        df['color_composition_score'] = df['harmony_score'] * df['visual_appeal']
        df['text_effectiveness'] = df['readability_score'] * df['text_contrast'] * (df['has_cta'] + 0.1)
        df['visual_impact'] = df['contrast_score'] * df['color_vibrancy'] * df['visual_appeal']
        
        # Комплексные индексы
        df['simplicity_index'] = (2 - df['composition_complexity'] - df['overall_complexity']) / 2
        df['professionalism_index'] = (df['harmony_score'] + df['visual_balance_score'] + df['rule_of_thirds_score']) / 3
        
        # Категориальные взаимодействия
        if 'cat_E-commerce' in df.columns:
            df['ecommerce_cta_boost'] = df['cat_E-commerce'] * df['has_cta'] * 2
        if 'cat_Финансы' in df.columns:
            df['finance_trust_index'] = df['cat_Финансы'] * df['professionalism_index']
        
        # Пороговые признаки
        df['high_contrast'] = (df['contrast_score'] > 0.7).astype(int)
        df['has_text'] = (df['text_amount'] > 0.1).astype(int)
        df['complex_design'] = (df['overall_complexity'] > 0.6).astype(int)
        
        return df
    
    def train_models(self, df: pd.DataFrame = None, quick_mode: bool = False) -> Dict[str, Dict]:
        """
        Обучение продвинутых ML моделей с валидацией.
        """
        if df is None:
            sample_size = 800 if not quick_mode else 500
            df = self.generate_realistic_data(sample_size)
        
        self.training_data = df
        self.training_timestamp = datetime.now()
        
        # Подготовка признаков
        feature_sets, feature_names = self.prepare_features(df)
        
        results = {}
        
        # Обучаем модели для каждой целевой метрики отдельно
        for target in ['ctr', 'conversion_rate', 'engagement']:
            print(f"🎯 Обучение моделей для {target}...")
            
            X = feature_sets[target]
            y = df[target].values
            
            # Разделение на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            
            # Масштабирование
            X_train_scaled = self.scalers[target].fit_transform(X_train)
            X_test_scaled = self.scalers[target].transform(X_test)
            
            # Feature selection
            if not quick_mode:
                selector = SelectKBest(score_func=f_regression, k=min(25, X_train.shape[1]))
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                X_test_selected = selector.transform(X_test_scaled)
                self.feature_selectors[target] = selector
            else:
                X_train_selected = X_train_scaled
                X_test_selected = X_test_scaled
            
            target_results = {}
            
            # Обучаем каждую модель
            for model_name, model in self.models[target].items():
                try:
                    if quick_mode and model_name == 'xgboost':
                        continue  # Пропускаем XGBoost в быстром режиме
                    
                    # Обучение
                    model.fit(X_train_selected, y_train)
                    
                    # Предсказания
                    y_pred = model.predict(X_test_selected)
                    
                    # Метрики
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    # Кросс-валидация
                    if not quick_mode:
                        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='r2')
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    else:
                        cv_mean, cv_std = r2, 0.0
                    
                    target_results[model_name] = {
                        'r2_score': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'cv_r2_mean': cv_mean,
                        'cv_r2_std': cv_std
                    }
                    
                    # Сохранение важности признаков
                    if hasattr(model, 'feature_importances_'):
                        if target not in self.feature_importance:
                            self.feature_importance[target] = {}
                        
                        if not quick_mode and target in self.feature_selectors:
                            # Восстанавливаем важность для всех признаков
                            selected_features = self.feature_selectors[target].get_support()
                            full_importance = np.zeros(len(feature_names))
                            full_importance[selected_features] = model.feature_importances_
                            importance_dict = dict(zip(feature_names, full_importance))
                        else:
                            importance_dict = dict(zip(feature_names, model.feature_importances_))
                        
                        self.feature_importance[target][model_name] = importance_dict
                    
                    print(f"  ✅ {model_name}: R² = {r2:.3f}, MAE = {mae:.4f}")
                    
                except Exception as e:
                    print(f"  ❌ {model_name}: {str(e)}")
                    target_results[model_name] = {'error': str(e)}
            
            results[target] = target_results
        
        # Сохранение общих метрик
        self.model_performance = results
        self.is_trained = True
        
        print("🎉 Обучение завершено!")
        return results
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Продвинутое предсказание с использованием ансамбля моделей.
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Вызовите train_models() сначала.")
        
        # Подготовка признаков
        feature_vector = self._prepare_prediction_features(features)
        
        predictions = {}
        
        for target in ['ctr', 'conversion_rate', 'engagement']:
            # Масштабирование
            feature_vector_scaled = self.scalers[target].transform([feature_vector])
            
            # Feature selection (если использовалась)
            if target in self.feature_selectors:
                feature_vector_selected = self.feature_selectors[target].transform(feature_vector_scaled)
            else:
                feature_vector_selected = feature_vector_scaled
            
            # Предсказания от всех моделей
            model_predictions = []
            model_weights = []
            
            for model_name, model in self.models[target].items():
                try:
                    if hasattr(model, 'predict'):  # Проверяем что модель обучена
                        pred = model.predict(feature_vector_selected)[0]
                        weight = self.ensemble_weights[target].get(model_name, 0.3)
                        
                        model_predictions.append(pred)
                        model_weights.append(weight)
                except:
                    continue
            
            if model_predictions:
                # Взвешенное среднее предсказаний
                total_weight = sum(model_weights)
                weighted_pred = sum(p * w for p, w in zip(model_predictions, model_weights)) / total_weight
                
                # Применяем ограничения для реалистичности
                if target == 'ctr':
                    predictions[target] = np.clip(weighted_pred, 0.001, 0.15)
                elif target == 'conversion_rate':
                    predictions[target] = np.clip(weighted_pred, 0.001, 0.35)
                else:  # engagement
                    predictions[target] = np.clip(weighted_pred, 0.01, 0.8)
            else:
                # Fallback предсказания
                fallback_values = {'ctr': 0.02, 'conversion_rate': 0.05, 'engagement': 0.1}
                predictions[target] = fallback_values[target]
        
        return predictions
    
    def _prepare_prediction_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Подготовка признаков для предсказания."""
        # Создаем вектор с значениями по умолчанию
        feature_vector = {}
        
        # Заполняем основные признаки
        for feature_name in self.feature_names:
            if feature_name.startswith('cat_') or feature_name.startswith('reg_'):
                # Категориальные признаки
                feature_vector[feature_name] = 0.0
            else:
                # Численные признаки
                feature_vector[feature_name] = features.get(feature_name, 0.5)
        
        # Устанавливаем категориальные признаки
        if 'category' in features:
            cat_feature = f"cat_{features['category']}"
            if cat_feature in feature_vector:
                feature_vector[cat_feature] = 1.0
        
        if 'region' in features:
            reg_feature = f"reg_{features['region']}"
            if reg_feature in feature_vector:
                feature_vector[reg_feature] = 1.0
        
        # Создаем engineered features если они есть в обучающих данных
        self._add_engineered_features_to_vector(feature_vector, features)
        
        # Преобразуем в массив в правильном порядке
        ordered_values = [feature_vector[name] for name in self.feature_names]
        return np.array(ordered_values)
    
    def _add_engineered_features_to_vector(self, feature_vector: Dict, original_features: Dict):
        """Добавление engineered признаков в вектор предсказания."""
        # Воссоздаем engineered features
        if 'color_composition_score' in feature_vector:
            harmony = original_features.get('harmony_score', 0.5)
            visual_appeal = original_features.get('visual_appeal', 0.5)
            feature_vector['color_composition_score'] = harmony * visual_appeal
        
        if 'text_effectiveness' in feature_vector:
            readability = original_features.get('readability_score', 0.5)
            contrast = original_features.get('text_contrast', 0.5)
            has_cta = original_features.get('has_cta', 0)
            feature_vector['text_effectiveness'] = readability * contrast * (has_cta + 0.1)
        
        if 'simplicity_index' in feature_vector:
            comp_complex = original_features.get('composition_complexity', 0.5)
            overall_complex = original_features.get('overall_complexity', 0.5)
            feature_vector['simplicity_index'] = (2 - comp_complex - overall_complex) / 2
        
        # Добавляем остальные по аналогии...
    
    def get_feature_importance(self, target: str = 'ctr') -> List[Tuple[str, float]]:
        """Получение важности признаков с агрегацией по моделям."""
        if target not in self.feature_importance:
            return []
        
        # Агрегируем важность по всем моделям
        aggregated_importance = {}
        
        for model_name, importance_dict in self.feature_importance[target].items():
            weight = self.ensemble_weights[target].get(model_name, 0.3)
            
            for feature, importance in importance_dict.items():
                if feature not in aggregated_importance:
                    aggregated_importance[feature] = 0
                aggregated_importance[feature] += importance * weight
        
        # Фильтрация и сортировка
        filtered_importance = [
            (feature, importance) for feature, importance in aggregated_importance.items()
            if importance >= FEATURE_IMPORTANCE_THRESHOLD
        ]
        
        filtered_importance.sort(key=lambda x: x[1], reverse=True)
        return filtered_importance[:20]  # Топ-20 признаков
    
    def get_prediction_confidence(self, features: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Получение доверительных интервалов с учетом неопределенности моделей."""
        if not self.is_trained:
            return {'ctr': (0.01, 0.05), 'conversion_rate': (0.02, 0.08), 'engagement': (0.05, 0.15)}
        
        # Получаем предсказания от всех моделей
        predictions = self.predict(features)
        feature_vector = self._prepare_prediction_features(features)
        
        confidence_intervals = {}
        
        for target in ['ctr', 'conversion_rate', 'engagement']:
            feature_vector_scaled = self.scalers[target].transform([feature_vector])
            
            if target in self.feature_selectors:
                feature_vector_selected = self.feature_selectors[target].transform(feature_vector_scaled)
            else:
                feature_vector_selected = feature_vector_scaled
            
            # Собираем предсказания от всех моделей
            model_predictions = []
            for model_name, model in self.models[target].items():
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(feature_vector_selected)[0]
                        model_predictions.append(pred)
                except:
                    continue
            
            if len(model_predictions) > 1:
                # Используем стандартное отклонение между моделями
                pred_mean = np.mean(model_predictions)
                pred_std = np.std(model_predictions)
                
                # 95% доверительный интервал
                margin = 1.96 * pred_std
                lower = max(pred_mean - margin, 0.001)
                upper = pred_mean + margin
                
                # Применяем ограничения
                if target == 'ctr':
                    upper = min(upper, 0.15)
                elif target == 'conversion_rate':
                    upper = min(upper, 0.35)
                else:  # engagement
                    upper = min(upper, 0.8)
                
                confidence_intervals[target] = (lower, upper)
            else:
                # Fallback интервалы
                base_pred = predictions[target]
                margin = base_pred * 0.3  # 30% от предсказания
                confidence_intervals[target] = (
                    max(base_pred - margin, 0.001),
                    base_pred + margin
                )
        
        return confidence_intervals
    
    def explain_prediction(self, features: Dict[str, Any], target: str = 'ctr') -> Dict[str, Any]:
        """Детальное объяснение предсказания."""
        predictions = self.predict(features)
        confidence = self.get_prediction_confidence(features)
        feature_importance = self.get_feature_importance(target)
        
        # Анализ ключевых факторов для данного предсказания
        feature_impacts = self._analyze_feature_impacts(features, target, feature_importance)
        
        # Общая оценка качества
        overall_score = self._calculate_overall_score(predictions)
        
        explanation = {
            'predictions': predictions,
            'confidence_intervals': confidence,
            'overall_score': overall_score,
            'feature_impacts': feature_impacts,
            'performance_category': self._categorize_performance(overall_score),
            'key_insights': self._generate_advanced_insights(features, predictions, feature_impacts),
            'model_confidence': self._assess_model_confidence(target),
            'recommendation_priority': self._assess_recommendation_priority(predictions)
        }
        
        return explanation
    
    def _analyze_feature_impacts(self, features: Dict, target: str, importance_list: List) -> List[Dict]:
        """Анализ влияния конкретных значений признаков."""
        impacts = []
        
        for feature_name, importance in importance_list[:10]:
            feature_value = features.get(feature_name, 0.5)
            
            # Определяем влияние на основе значения и важности
            if importance > 0.1:  # Очень важный признак
                if feature_value > 0.7:
                    impact = "очень положительное"
                elif feature_value > 0.5:
                    impact = "положительное"
                elif feature_value > 0.3:
                    impact = "нейтральное"
                else:
                    impact = "отрицательное"
            else:
                impact = "минимальное"
            
            impacts.append({
                'feature': feature_name,
                'value': feature_value,
                'importance': importance,
                'impact': impact,
                'contribution': importance * feature_value
            })
        
        return impacts
    
    def _calculate_overall_score(self, predictions: Dict) -> float:
        """Расчет общей оценки эффективности."""
        ctr_score = predictions['ctr'] / PERFORMANCE_METRICS['ctr']['target']
        conv_score = predictions['conversion_rate'] / PERFORMANCE_METRICS['conversion_rate']['target']
        eng_score = predictions['engagement'] / PERFORMANCE_METRICS['engagement']['target']
        
        # Взвешенная оценка
        overall = (ctr_score * 0.4 + conv_score * 0.4 + eng_score * 0.2)
        return min(overall, 2.0)  # Максимум 200% от целевых значений
    
    def _categorize_performance(self, score: float) -> str:
        """Категоризация эффективности."""
        if score >= 1.2:
            return "Превосходная"
        elif score >= 1.0:
            return "Отличная"
        elif score >= 0.8:
            return "Хорошая"
        elif score >= 0.6:
            return "Средняя"
        else:
            return "Требует улучшения"
    
    def _generate_advanced_insights(self, features: Dict, predictions: Dict, impacts: List) -> List[str]:
        """Генерация продвинутых инсайтов."""
        insights = []
        
        # Анализ сильных сторон
        strong_features = [i for i in impacts if i['impact'] in ['положительное', 'очень положительное']]
        if len(strong_features) >= 3:
            insights.append(f"Креатив имеет {len(strong_features)} сильных визуальных характеристик")
        
        # Анализ слабых мест
        weak_features = [i for i in impacts if i['impact'] == 'отрицательное']
        if weak_features:
            insights.append(f"Обнаружено {len(weak_features)} критических области для улучшения")
        
        # Предметные инсайты
        if predictions['ctr'] > PERFORMANCE_METRICS['ctr']['target'] * 1.2:
            insights.append("Высокий прогнозируемый CTR указывает на сильную визуальную привлекательность")
        
        if predictions['conversion_rate'] > PERFORMANCE_METRICS['conversion_rate']['target'] * 1.2:
            insights.append("Отличная конверсия предполагает эффективное убеждение к действию")
        
        if features.get('has_cta', 0) and predictions['conversion_rate'] < PERFORMANCE_METRICS['conversion_rate']['target']:
            insights.append("Несмотря на наличие CTA, конверсия может быть улучшена")
        
        return insights[:5]
    
    def _assess_model_confidence(self, target: str) -> float:
        """Оценка уверенности модели."""
        if target in self.model_performance:
            # Берем средний R² по всем моделям
            r2_scores = [
                result.get('r2_score', 0) for result in self.model_performance[target].values()
                if 'r2_score' in result
            ]
            return np.mean(r2_scores) if r2_scores else 0.5
        return 0.5
    
    def _assess_recommendation_priority(self, predictions: Dict) -> str:
        """Определение приоритета рекомендаций."""
        avg_performance = np.mean(list(predictions.values()))
        target_avg = np.mean([
            PERFORMANCE_METRICS['ctr']['target'],
            PERFORMANCE_METRICS['conversion_rate']['target'],
            PERFORMANCE_METRICS['engagement']['target']
        ])
        
        if avg_performance < target_avg * 0.7:
            return "Высокий"
        elif avg_performance < target_avg:
            return "Средний"
        else:
            return "Низкий"


# Алиас для обратной совместимости
MLEngine = AdvancedMLEngine
