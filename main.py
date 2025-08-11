# main.py
"""
Основное приложение Creative Performance Predictor.
Streamlit интерфейс для анализа и оптимизации креативов.
"""

# main.py
"""
Основное приложение Creative Performance Predictor.
Streamlit интерфейс для анализа и оптимизации креативов.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import time
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Проверка доступности зависимостей
missing_deps = []

try:
    from image_analyzer import ImageAnalyzer
except ImportError as e:
    missing_deps.append(f"Image Analyzer: {str(e)}")
    ImageAnalyzer = None

try:
    from ml_engine import MLEngine
except ImportError as e:
    missing_deps.append(f"ML Engine: {str(e)}")
    MLEngine = None

try:
    from visualizer import Visualizer
except ImportError as e:
    missing_deps.append(f"Visualizer: {str(e)}")
    Visualizer = None

try:
    from recommender import RecommendationEngine
except ImportError as e:
    missing_deps.append(f"Recommender: {str(e)}")
    RecommendationEngine = None

try:
    from config import (
        APP_TITLE, APP_VERSION, PAGE_ICON, SUPPORTED_IMAGE_FORMATS,
        MAX_IMAGE_SIZE, CUSTOM_CSS, DEMO_INSIGHTS, COLOR_SCHEME
    )
except ImportError as e:
    missing_deps.append(f"Config: {str(e)}")
    # Fallback значения
    APP_TITLE = "Creative Performance Predictor"
    APP_VERSION = "1.0.0"
    PAGE_ICON = "🎨"
    SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png']
    MAX_IMAGE_SIZE = 10 * 1024 * 1024
    CUSTOM_CSS = ""
    DEMO_INSIGHTS = ["Система готова к анализу ваших креативов!"]
    COLOR_SCHEME = {}

# Конфигурация страницы
st.set_page_config(
    page_title=f"{APP_TITLE} v{APP_VERSION}",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Применение кастомных стилей
if CUSTOM_CSS:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

class CreativePerformanceApp:
    """Главный класс приложения Creative Performance Predictor."""
    
    def __init__(self):
        """Инициализация приложения и компонентов."""
        if not all([ImageAnalyzer, MLEngine, Visualizer, RecommendationEngine]):
            st.error("Не удается инициализировать приложение - отсутствуют критические компоненты")
            return
            
        self.analyzer = ImageAnalyzer()
        self.ml_engine = MLEngine()
        self.visualizer = Visualizer()
        self.recommender = RecommendationEngine()
        
        # Инициализация состояния сессии
        self._initialize_session_state()
    
    def _try_load_pretrained_model(self):
        """Попытка загрузить предобученную модель."""
        try:
            import pickle
            import os
            
            if os.path.exists('quick_model.pkl'):
                with open('quick_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                # Загружаем данные в ML engine
                if 'models' in model_data:
                    self.ml_engine.models.update(model_data['models'])
                if 'scalers' in model_data:
                    self.ml_engine.scalers.update(model_data['scalers'])
                if 'feature_names' in model_data:
                    self.ml_engine.feature_names = model_data['feature_names']
                if 'is_trained' in model_data:
                    self.ml_engine.is_trained = model_data['is_trained']
                    st.session_state.model_trained = True
                    
                # Показываем уведомление только один раз
                if 'pretrained_loaded' not in st.session_state:
                    st.session_state.pretrained_loaded = True
                    st.info("ℹ️ Загружена предобученная модель")
        
        except Exception as e:
            # Тихо игнорируем ошибки загрузки предобученной модели
            pass
    
    def _initialize_session_state(self):
        """Инициализация переменных состояния сессии."""
        session_defaults = {
            'model_trained': False,
            'image_uploaded': False,
            'analysis_completed': False,
            'current_image': None,
            'image_features': {},
            'predictions': {},
            'recommendations': [],
            'analysis_results': {}
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def _train_model(self):
        """Обучение ML модели."""
        try:
            # Принудительно инициализируем модели заново
            self.ml_engine._initialize_models()
            
            with st.spinner('🤖 Обучение модели машинного обучения...'):
                # Простое обучение без сложных проверок
                training_results = self.ml_engine.train_models(quick_mode=True)
                
                # Устанавливаем состояние как обученное
                self.ml_engine.is_trained = True
                st.session_state.model_trained = True
                st.session_state.training_results = training_results
                
                st.success("✅ Модель успешно обучена!")
                
                # Показываем статистику
                with st.expander("📊 Статистика обучения", expanded=False):
                    st.write(f"**Количество признаков:** {len(self.ml_engine.feature_names)}")
                    for target, models in training_results.items():
                        st.write(f"**{target.upper()}:**")
                        for model_name, metrics in models.items():
                            r2 = metrics.get('r2_score', 0)
                            st.write(f"  - {model_name}: R² = {r2:.3f}")
            
        except Exception as e:
            st.error(f"❌ Ошибка при обучении модели: {str(e)}")
            st.session_state.model_trained = False
            if hasattr(self, 'ml_engine'):
                self.ml_engine.is_trained = False
            
            with st.expander("🔍 Детали ошибки"):
                st.code(str(e))
            
            if st.button("🔄 Повторить обучение модели"):
                st.rerun()
    
    def run(self):
        """Запуск основного приложения."""
        # Заголовок приложения
        st.markdown(f'<h1 class="main-header">{PAGE_ICON} {APP_TITLE}</h1>', 
                   unsafe_allow_html=True)
        
        # Боковая панель с навигацией
        self._render_sidebar()
        
        # Основной контент
        page = st.session_state.get('current_page', 'Главная')
        
        if page == 'Главная':
            self._render_home_page()
        elif page == 'Анализ изображения':
            self._render_analysis_page()
        elif page == 'Результаты':
            self._render_results_page()
        elif page == 'Рекомендации':
            self._render_recommendations_page()
        elif page == 'О проекте':
            self._render_about_page()
    
    def _render_sidebar(self):
        """Отрисовка боковой панели."""
        with st.sidebar:
            st.markdown("### 🧭 Навигация")
            
            # Кнопки навигации
            pages = [
                ('🏠', 'Главная'),
                ('🔍', 'Анализ изображения'),
                ('📊', 'Результаты'),
                ('💡', 'Рекомендации'),
                ('ℹ️', 'О проекте')
            ]
            
            for icon, page_name in pages:
                if st.button(f"{icon} {page_name}", key=f"nav_{page_name}"):
                    st.session_state.current_page = page_name
                    st.rerun()
            
            st.markdown("---")
            
            # Статус системы
            st.markdown("### 📈 Статус системы")
            
            # Статус модели
            if st.session_state.model_trained:
                st.success("✅ Модель обучена")
            else:
                st.error("❌ Модель не обучена")
                if st.button("🔄 Обучить модель", key="retrain_sidebar"):
                    self._train_model()
                    st.rerun()
            
            # Статус изображения
            if st.session_state.image_uploaded:
                st.success("✅ Изображение загружено")
            else:
                st.info("📤 Загрузите изображение")
            
            # Статус анализа
            if st.session_state.analysis_completed:
                st.success("✅ Анализ завершен")
            else:
                st.info("🔄 Анализ не выполнен")
            
            st.markdown("---")
            
            # Демо-инсайты
            st.markdown("### 💭 Знаете ли вы?")
            insight = np.random.choice(DEMO_INSIGHTS)
            st.info(insight)
            
            # Информация о версии
            st.markdown("---")
            st.caption(f"Версия {APP_VERSION}")
    
    def _render_home_page(self):
        """Отрисовка главной страницы."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## 🎨 Добро пожаловать в Creative Performance Predictor!
            
            Это интеллектуальная система для анализа и оптимизации креативных материалов 
            с использованием компьютерного зрения и машинного обучения.
            
            ### 🔍 Что умеет система:
            
            **Анализ изображений**
            - Извлечение цветовых характеристик
            - Анализ композиции и баланса
            - Оценка текстовых элементов
            
            **Предсказание эффективности**
            - Прогноз CTR (Click-Through Rate)
            - Оценка конверсий
            - Прогноз вовлеченности аудитории
            
            **Персонализированные рекомендации**
            - Конкретные советы по улучшению
            - Приоритизация изменений
            - Пошаговые планы действий
            """)
            
            # Кнопка быстрого старта
            if st.session_state.model_trained:
                if st.button("🚀 Начать анализ", type="primary", key="quick_start"):
                    st.session_state.current_page = 'Анализ изображения'
                    st.rerun()
            else:
                if st.button("🤖 Обучить модель", type="primary", key="train_model_main"):
                    self._train_model()
                    st.rerun()
        
        with col2:
            st.markdown("### 📊 Возможности системы")
            
            # Метрики возможностей
            metrics_data = [
                ("🎯", "Точность", "R² > 0.85"),
                ("⚡", "Скорость", "< 5 сек"),
                ("🔧", "Рекомендации", "10+ советов"),
                ("📱", "Форматы", "JPG, PNG, WEBP")
            ]
            
            for icon, metric, value in metrics_data:
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{icon} {metric}</h4>
                        <p>{value}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Пример анализа
            st.markdown("### 🖼️ Пример анализа")
            st.image("https://via.placeholder.com/300x200/667eea/ffffff?text=Demo+Creative", 
                    caption="Демо-креатив для анализа")
            
            if st.button("📋 Посмотреть демо"):
                st.info("Загрузите собственное изображение для полного анализа!")
    
    def _render_analysis_page(self):
        """Отрисовка страницы анализа изображения."""
        st.header("🔍 Анализ креативного материала")
        
        # Загрузка изображения
        uploaded_file = st.file_uploader(
            "Загрузите изображение для анализа",
            type=SUPPORTED_IMAGE_FORMATS,
            help=f"Поддерживаемые форматы: {', '.join(SUPPORTED_IMAGE_FORMATS)}. Максимальный размер: {MAX_IMAGE_SIZE // (1024*1024)}MB"
        )
        
        if uploaded_file is not None:
            # Проверка размера файла
            if uploaded_file.size > MAX_IMAGE_SIZE:
                st.error(f"Размер файла превышает {MAX_IMAGE_SIZE // (1024*1024)}MB")
                return
            
            # Загрузка и отображение изображения
            try:
                image = Image.open(uploaded_file)
                st.session_state.current_image = image
                st.session_state.image_uploaded = True
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(image, caption="Загруженное изображение", use_container_width=True)
                
                with col2:
                    st.markdown("### 📏 Характеристики изображения")
                    st.write(f"**Размер:** {image.size[0]} × {image.size[1]} пикселей")
                    st.write(f"**Формат:** {image.format}")
                    st.write(f"**Режим:** {image.mode}")
                    st.write(f"**Размер файла:** {uploaded_file.size / 1024:.1f} KB")
                
                # Кнопка запуска анализа
                if st.session_state.model_trained:
                    if st.button("🚀 Запустить анализ", type="primary", key="start_analysis"):
                        self._perform_analysis(image)
                else:
                    st.error("❌ Модель не обучена.")
                    if st.button("🤖 Обучить модель", type="primary", key="train_before_analysis"):
                        self._train_model()
                        st.rerun()
                
            except Exception as e:
                st.error(f"Ошибка при загрузке изображения: {str(e)}")
        
        else:
            # Информация о том, как загрузить изображение
            st.info("""
            👆 **Загрузите изображение креатива для начала анализа**
            
            Система проанализирует:
            - Цветовые характеристики
            - Композиционные элементы  
            - Текстовое содержание
            - Предскажет эффективность
            - Сгенерирует рекомендации
            """)
            
            # Демо-изображение
            if st.button("🎲 Загрузить демо-изображение"):
                # Создание простого демо-изображения
                demo_image = self._create_demo_image()
                if demo_image:
                    st.session_state.current_image = demo_image
                    st.session_state.image_uploaded = True
                    st.success("✅ Демо-изображение загружено!")
                    st.rerun()
                else:
                    st.error("❌ Не удалось создать демо-изображение")
    
    def _perform_analysis(self, image: Image.Image):
        """Выполнение полного анализа изображения."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Этап 1: Загрузка изображения в анализатор
            status_text.text("🔄 Подготовка изображения...")
            progress_bar.progress(10)
            
            if not self.analyzer.load_image(image):
                st.error("Не удалось загрузить изображение для анализа")
                return
            
            time.sleep(0.5)  # Для демонстрации прогресса
            
            # Этап 2: Анализ цветовых характеристик
            status_text.text("🎨 Анализ цветовых характеристик...")
            progress_bar.progress(30)
            
            color_analysis = self.analyzer.analyze_colors()
            time.sleep(0.5)
            
            # Этап 3: Анализ композиции
            status_text.text("📐 Анализ композиции...")
            progress_bar.progress(50)
            
            composition_analysis = self.analyzer.analyze_composition()
            time.sleep(0.5)
            
            # Этап 4: Анализ текста
            status_text.text("📝 Анализ текстовых элементов...")
            progress_bar.progress(70)
            
            text_analysis = self.analyzer.analyze_text()
            time.sleep(0.5)
            
            # Этап 5: Извлечение признаков для ML
            status_text.text("🧠 Подготовка данных для ML...")
            progress_bar.progress(80)
            
            image_features = self.analyzer.get_all_features()
            st.session_state.image_features = image_features
            
            # Этап 6: Предсказание эффективности
            status_text.text("🔮 Предсказание эффективности...")
            progress_bar.progress(90)
            
            predictions = self.ml_engine.predict(image_features)
            confidence_intervals = self.ml_engine.get_prediction_confidence(image_features)
            
            # Этап 7: Генерация рекомендаций
            status_text.text("💡 Генерация рекомендаций...")
            progress_bar.progress(100)
            
            recommendations = self.recommender.generate_recommendations(
                image_features, predictions
            )
            
            # Сохранение результатов
            st.session_state.analysis_results = {
                'color_analysis': color_analysis,
                'composition_analysis': composition_analysis,
                'text_analysis': text_analysis,
                'image_features': image_features,
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'recommendations': recommendations
            }
            
            st.session_state.predictions = predictions
            st.session_state.recommendations = recommendations
            st.session_state.analysis_completed = True
            
            # Завершение
            status_text.text("✅ Анализ завершен!")
            time.sleep(1)
            
            # Очистка индикаторов прогресса
            progress_bar.empty()
            status_text.empty()
            
            # Показ результатов
            st.success("🎉 Анализ успешно завершен! Переходите к результатам.")
            
            if st.button("📊 Посмотреть результаты", type="primary"):
                st.session_state.current_page = 'Результаты'
                st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Ошибка при анализе: {str(e)}")
    
    def _render_results_page(self):
        """Отрисовка страницы с результатами анализа."""
        if not st.session_state.analysis_completed:
            st.warning("⚠️ Сначала выполните анализ изображения")
            if st.button("🔍 Перейти к анализу"):
                st.session_state.current_page = 'Анализ изображения'
                st.rerun()
            return
        
        st.header("📊 Результаты анализа креатива")
        
        # Краткий обзор результатов
        col1, col2, col3, col4 = st.columns(4)
        
        predictions = st.session_state.predictions
        
        with col1:
            ctr_value = predictions.get('ctr', 0) * 100
            st.metric(
                "CTR", 
                f"{ctr_value:.2f}%",
                delta=f"{ctr_value - 2.0:.2f}% от цели"
            )
        
        with col2:
            conv_value = predictions.get('conversion_rate', 0) * 100
            st.metric(
                "Конверсия", 
                f"{conv_value:.2f}%",
                delta=f"{conv_value - 5.0:.2f}% от цели"
            )
        
        with col3:
            eng_value = predictions.get('engagement', 0) * 100
            st.metric(
                "Вовлеченность", 
                f"{eng_value:.2f}%",
                delta=f"{eng_value - 10.0:.2f}% от цели"
            )
        
        with col4:
            # Общая оценка
            overall_score = (ctr_value/2.0 + conv_value/5.0 + eng_value/10.0) / 3 * 100
            st.metric(
                "Общая оценка", 
                f"{overall_score:.0f}/100",
                delta=f"{'Отлично' if overall_score > 80 else 'Хорошо' if overall_score > 60 else 'Требует улучшения'}"
            )
        
        st.markdown("---")
        
        # Вкладки с детальными результатами
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎨 Цветовой анализ",
            "📐 Композиция", 
            "📝 Текст",
            "🔮 Предсказания"
        ])
        
        analysis_results = st.session_state.analysis_results
        
        with tab1:
            st.subheader("Анализ цветовых характеристик")
            
            if 'color_analysis' in analysis_results:
                color_data = analysis_results['color_analysis']
                
                # График цветового анализа
                color_fig = self.visualizer.plot_color_analysis(color_data)
                st.plotly_chart(color_fig, use_container_width=True)
                
                # Детальная информация
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Ключевые метрики:**")
                    st.write(f"• Яркость: {color_data.get('brightness', 0):.2f}")
                    st.write(f"• Насыщенность: {color_data.get('saturation', 0):.2f}")
                    st.write(f"• Контрастность: {color_data.get('contrast_score', 0):.2f}")
                    st.write(f"• Цветовая гармония: {color_data.get('harmony_score', 0):.2f}")
                
                with col2:
                    st.markdown("**Цветовая палитра:**")
                    if 'dominant_colors' in color_data:
                        colors = color_data['dominant_colors'][:5]
                        for i, color in enumerate(colors):
                            color_name = f"rgb({color[0]},{color[1]},{color[2]})"
                            st.markdown(
                                f"<div style='background-color: {color_name}; "
                                f"padding: 10px; margin: 2px; border-radius: 5px;'>"
                                f"Цвет {i+1}: {color_name}</div>",
                                unsafe_allow_html=True
                            )
        
        with tab2:
            st.subheader("Анализ композиции")
            
            if 'composition_analysis' in analysis_results:
                comp_data = analysis_results['composition_analysis']
                
                # График композиционного анализа
                comp_fig = self.visualizer.plot_composition_analysis(comp_data)
                st.plotly_chart(comp_fig, use_container_width=True)
                
                # Детальная информация
                st.markdown("**Композиционные характеристики:**")
                
                metrics = [
                    ("Правило третей", comp_data.get('rule_of_thirds_score', 0)),
                    ("Визуальный баланс", comp_data.get('visual_balance_score', 0)),
                    ("Сложность композиции", comp_data.get('composition_complexity', 0)),
                    ("Центральный фокус", comp_data.get('center_focus_score', 0)),
                ]
                
                for metric_name, value in metrics:
                    progress_color = "green" if value > 0.7 else "orange" if value > 0.4 else "red"
                    st.markdown(
                        f"**{metric_name}:** {value:.2f} "
                        f"<div style='width: 100%; background-color: #f0f0f0; border-radius: 10px;'>"
                        f"<div style='width: {value*100}%; background-color: {progress_color}; "
                        f"height: 20px; border-radius: 10px;'></div></div>",
                        unsafe_allow_html=True
                    )
        
        with tab3:
            st.subheader("Анализ текстовых элементов")
            
            if 'text_analysis' in analysis_results:
                text_data = analysis_results['text_analysis']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Характеристики текста:**")
                    st.write(f"• Количество текстовых блоков: {text_data.get('text_amount', 0)}")
                    st.write(f"• Общее количество символов: {text_data.get('total_characters', 0)}")
                    st.write(f"• Читаемость: {text_data.get('readability_score', 0):.2f}")
                    st.write(f"• Иерархия: {text_data.get('text_hierarchy', 0):.2f}")
                
                with col2:
                    st.markdown("**Качество текста:**")
                    st.write(f"• Позиционирование: {text_data.get('text_positioning', 0):.2f}")
                    st.write(f"• Контрастность: {text_data.get('text_contrast', 0):.2f}")
                    
                    has_cta = text_data.get('has_cta', False)
                    cta_status = "✅ Есть" if has_cta else "❌ Отсутствует"
                    st.write(f"• Призыв к действию: {cta_status}")
        
        with tab4:
            st.subheader("Предсказания эффективности")
            
            # График предсказаний
            pred_fig = self.visualizer.plot_performance_prediction(
                predictions,
                analysis_results.get('confidence_intervals')
            )
            st.plotly_chart(pred_fig, use_container_width=True)
            
            # График важности признаков
            if st.session_state.model_trained:
                feature_importance = self.ml_engine.get_feature_importance('ctr')
                if feature_importance:
                    importance_fig = self.visualizer.plot_feature_importance(feature_importance)
                    st.plotly_chart(importance_fig, use_container_width=True)
            
            # Объяснение предсказаний
            if st.button("🔍 Подробное объяснение предсказаний"):
                with st.expander("Объяснение модели", expanded=True):
                    explanation = self.ml_engine.explain_prediction(
                        st.session_state.image_features
                    )
                    
                    st.markdown("**Ключевые факторы влияния:**")
                    for impact in explanation.get('feature_impacts', [])[:5]:
                        st.write(f"• {impact['feature']}: {impact['impact']} влияние")
                    
                    st.markdown("**Инсайты:**")
                    for insight in explanation.get('key_insights', []):
                        st.write(f"• {insight}")
        
        # Переход к рекомендациям
        st.markdown("---")
        if st.button("💡 Посмотреть рекомендации", type="primary"):
            st.session_state.current_page = 'Рекомендации'
            st.rerun()
    
    def _render_recommendations_page(self):
        """Отрисовка страницы рекомендаций."""
        if not st.session_state.analysis_completed:
            st.warning("⚠️ Сначала выполните анализ изображения")
            return
        
        st.header("💡 Рекомендации по улучшению креатива")
        
        recommendations = st.session_state.recommendations
        
        if not recommendations:
            st.info("Рекомендации не найдены")
            return
        
        # Сводка рекомендаций
        high_priority = [r for r in recommendations if r.priority == 'high']
        medium_priority = [r for r in recommendations if r.priority == 'medium']
        low_priority = [r for r in recommendations if r.priority == 'low']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Всего рекомендаций", len(recommendations))
        with col2:
            st.metric("Высокий приоритет", len(high_priority))
        with col3:
            st.metric("Средний приоритет", len(medium_priority))
        with col4:
            total_impact = sum(r.expected_impact for r in recommendations)
            st.metric("Потенциальное улучшение", f"{total_impact:.1%}")
        
        st.markdown("---")
        
        # Вкладки с рекомендациями по категориям
        tab1, tab2, tab3 = st.tabs([
            "🔥 Приоритетные",
            "📋 Все рекомендации",
            "📈 План действий"
        ])
        
        with tab1:
            st.subheader("Рекомендации высокого приоритета")
            
            if high_priority:
                for i, rec in enumerate(high_priority, 1):
                    with st.container():
                        st.markdown(
                            f"""
                            <div class="recommendation-high">
                                <h4>🔥 {rec.title}</h4>
                                <p><strong>Описание:</strong> {rec.description}</p>
                                <p><strong>Ожидаемое улучшение:</strong> {rec.expected_impact:.1%}</p>
                                <p><strong>Уверенность:</strong> {rec.confidence:.1%}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        with st.expander(f"Шаги для выполнения рекомендации {i}"):
                            for j, step in enumerate(rec.actionable_steps, 1):
                                st.write(f"{j}. {step}")
                        
                        st.markdown("---")
            else:
                st.success("🎉 Нет критических проблем! Креатив хорошо оптимизирован.")
        
        with tab2:
            st.subheader("Все рекомендации")
            
            # Фильтры
            col1, col2 = st.columns(2)
            
            with col1:
                priority_filter = st.multiselect(
                    "Фильтр по приоритету",
                    ['high', 'medium', 'low'],
                    default=['high', 'medium', 'low'],
                    format_func=lambda x: {'high': 'Высокий', 'medium': 'Средний', 'low': 'Низкий'}[x]
                )
            
            with col2:
                category_filter = st.multiselect(
                    "Фильтр по категории",
                    list(set(r.category for r in recommendations)),
                    default=list(set(r.category for r in recommendations)),
                    format_func=lambda x: {
                        'color': 'Цвет', 'composition': 'Композиция', 
                        'text': 'Текст', 'overall': 'Общее'
                    }.get(x, x)
                )
            
            # Отфильтрованные рекомендации
            filtered_recs = [
                r for r in recommendations 
                if r.priority in priority_filter and r.category in category_filter
            ]
            
            for i, rec in enumerate(filtered_recs, 1):
                priority_class = f"recommendation-{rec.priority}"
                priority_emoji = {'high': '🔥', 'medium': '⚡', 'low': '💡'}[rec.priority]
                
                with st.container():
                    st.markdown(
                        f"""
                        <div class="{priority_class}">
                            <h4>{priority_emoji} {rec.title}</h4>
                            <p>{rec.description}</p>
                            <small>Категория: {rec.category} | Улучшение: {rec.expected_impact:.1%} | 
                            Уверенность: {rec.confidence:.1%}</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    with st.expander(f"Подробности рекомендации {i}"):
                        st.markdown("**Пошаговые действия:**")
                        for j, step in enumerate(rec.actionable_steps, 1):
                            st.write(f"{j}. {step}")
        
        with tab3:
            st.subheader("План действий")
            
            action_plan = self.recommender.create_action_plan(recommendations)
            
            # Сводка плана
            summary = action_plan['summary']
            
            st.markdown(f"""
            **📊 Сводка:**
            - Всего рекомендаций: {summary['total_recommendations']}
            - Высокоприоритетных: {summary['high_priority_count']}
            - Потенциальное улучшение: {summary['potential_improvement']}
            - Ориентировочное время: {summary['estimated_total_time']}
            """)
            
            # Немедленные действия
            if action_plan['immediate_actions']:
                st.markdown("### 🚨 Немедленные действия (1-2 часа)")
                for action in action_plan['immediate_actions']:
                    with st.expander(f"⚡ {action['title']} (Влияние: {action['expected_impact']:.1%})"):
                        st.write(action['description'])
                        st.markdown("**Первые шаги:**")
                        for step in action['steps']:
                            st.write(f"• {step}")
            
            # Краткосрочные действия
            if action_plan['short_term_actions']:
                st.markdown("### 📅 Краткосрочные действия (2-4 часа)")
                for action in action_plan['short_term_actions']:
                    with st.expander(f"🔧 {action['title']} (Влияние: {action['expected_impact']:.1%})"):
                        st.write(action['description'])
                        st.markdown("**Действия:**")
                        for step in action['steps']:
                            st.write(f"• {step}")
            
            # Долгосрочные улучшения
            if action_plan['long_term_improvements']:
                st.markdown("### 🔮 Долгосрочные улучшения (4+ часов)")
                for action in action_plan['long_term_improvements']:
                    with st.expander(f"🎯 {action['title']} (Влияние: {action['expected_impact']:.1%})"):
                        st.write(action['description'])
    
    def _render_about_page(self):
        """Отрисовка страницы 'О проекте'."""
        st.header("ℹ️ О проекте Creative Performance Predictor")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            ## 🎯 Миссия проекта
            
            Creative Performance Predictor - это инновационная система, которая использует 
            современные технологии компьютерного зрения и машинного обучения для анализа 
            и оптимизации креативных материалов.
            
            ## 🔬 Научная основа
            
            Система базируется на следующих принципах:
            
            **Компьютерное зрение**
            - Извлечение цветовых характеристик через анализ HSV пространства
            - Композиционный анализ с использованием правила третей
            - OCR для анализа текстовых элементов
            
            **Машинное обучение**
            - Random Forest для интерпретируемых предсказаний
            - XGBoost для максимальной точности
            - Ансамблевые методы для повышения надежности
            
            **Маркетинговая аналитика**
            - Предсказание CTR, конверсий и вовлеченности
            - Анализ важности признаков
            - Генерация персонализированных рекомендаций
            
            ## 🛠 Технологический стек
            
            - **Python** - основной язык разработки
            - **Streamlit** - веб-интерфейс
            - **OpenCV** - компьютерное зрение
            - **Scikit-learn** - машинное обучение
            - **XGBoost** - градиентный бустинг
            - **Plotly** - интерактивная визуализация
            
            ## 📈 Версия {APP_VERSION}
            
            Текущая версия включает:
            - Полный анализ изображений
            - Предсказание эффективности
            - Персонализированные рекомендации
            - Интерактивные визуализации
            """)
        
        with col2:
            st.markdown("### 📊 Статистика модели")
            
            if st.session_state.model_trained and 'training_results' in st.session_state:
                training_results = st.session_state.training_results
                
                # Показать метрики обучения
                for target, models in training_results.items():
                    st.markdown(f"**{target.upper()}:**")
                    for model_name, metrics in models.items():
                        r2 = metrics.get('r2_score', 0)
                        st.write(f"• {model_name}: R² = {r2:.3f}")
                    st.write("")
            
            st.markdown("### 🎨 Примеры анализа")
            
            examples = [
                "Автомобильная реклама",
                "E-commerce баннер", 
                "Финансовые услуги",
                "Технологические продукты"
            ]
            
            for example in examples:
                st.write(f"• {example}")
            
            st.markdown("### 🔗 Полезные ссылки")
            
            st.markdown("""
            - [Документация по API](https://docs.example.com)
            - [GitHub репозиторий](https://github.com/user/cpp)
            - [Обратная связь](mailto:feedback@example.com)
            """)
        
        # Техническая информация
        st.markdown("---")
        st.markdown("### 🔧 Техническая информация")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Поддерживаемые форматы:**
            - JPG/JPEG
            - PNG  
            - WEBP
            - BMP
            """)
        
        with col2:
            st.markdown("""
            **Ограничения:**
            - Максимальный размер: 10MB
            - Минимальное разрешение: 100x100
            - Максимальное разрешение: 8192x8192
            """)
        
        with col3:
            st.markdown("""
            **Производительность:**
            - Время анализа: < 5 секунд
            - Точность модели: R² > 0.85
            - Количество признаков: 20+
            """)
    
    def _create_demo_image(self) -> Optional[Image.Image]:
        """Создание демонстрационного изображения."""
        try:
            from PIL import ImageDraw, ImageFont
            import numpy as np
            
            # Создание изображения с градиентом
            width, height = 800, 600
            image = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(image)
            
            # Создание градиентного фона (желтый как у Максима)
            for y in range(height):
                # Градиент от ярко-желтого к оранжевому
                color_intensity = int(255 * (1 - y / height * 0.3))
                color = (255, color_intensity, 0)  # Желто-оранжевый градиент
                draw.line([(0, y), (width, y)], fill=color)
            
            # Добавление текста
            try:
                # Попытка использовать системный шрифт
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
            except:
                font_large = font_medium = None
            
            # Основной заголовок
            text_main = "DEMO CREATIVE"
            
            # Современный способ получения размера текста
            try:
                # Для новых версий Pillow
                bbox = draw.textbbox((0, 0), text_main, font=font_large)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except AttributeError:
                # Для старых версий Pillow
                try:
                    text_w, text_h = draw.textsize(text_main, font=font_large)
                except:
                    text_w, text_h = 200, 40
                    
            x = (width - text_w) // 2
            y = height // 3
            
            # Тень для текста
            draw.text((x+3, y+3), text_main, fill=(50, 50, 50), font=font_large)
            # Основной текст
            draw.text((x, y), text_main, fill=(255, 255, 255), font=font_large)
            
            # Подзаголовок
            text_sub = "Test Image for Analysis"
            try:
                bbox2 = draw.textbbox((0, 0), text_sub, font=font_medium)
                text_w2 = bbox2[2] - bbox2[0]
                text_h2 = bbox2[3] - bbox2[1]
            except AttributeError:
                try:
                    text_w2, text_h2 = draw.textsize(text_sub, font=font_medium)
                except:
                    text_w2, text_h2 = 150, 20
                    
            x2 = (width - text_w2) // 2
            y2 = y + text_h + 20
            draw.text((x2, y2), text_sub, fill=(100, 100, 100), font=font_medium)
            
            # Добавление простых геометрических элементов
            # Круг в правом верхнем углу
            circle_x, circle_y = width - 150, 100
            draw.ellipse([circle_x-50, circle_y-50, circle_x+50, circle_y+50], 
                        fill=(255, 100, 100), outline=(200, 50, 50), width=3)
            
            # Прямоугольник в левом нижнем углу
            rect_x, rect_y = 100, height - 150
            draw.rectangle([rect_x-40, rect_y-30, rect_x+40, rect_y+30], 
                          fill=(100, 150, 255), outline=(50, 100, 200), width=3)
            
            # CTA кнопка
            button_x, button_y = width // 2, height - 100
            button_w, button_h = 120, 40
            draw.rectangle([button_x-button_w//2, button_y-button_h//2, 
                           button_x+button_w//2, button_y+button_h//2], 
                          fill=(220, 50, 50), outline=(180, 30, 30), width=2)
            
            cta_text = "CLICK HERE"
            try:
                # Современный способ для новых версий Pillow
                try:
                    bbox3 = draw.textbbox((0, 0), cta_text, font=font_medium)
                    cta_w = bbox3[2] - bbox3[0]
                    cta_h = bbox3[3] - bbox3[1]
                except AttributeError:
                    # Для старых версий Pillow
                    cta_w, cta_h = draw.textsize(cta_text, font=font_medium) if font_medium else (80, 15)
                    
                draw.text((button_x - cta_w//2, button_y - cta_h//2), 
                         cta_text, fill=(255, 255, 255), font=font_medium)
            except:
                pass
            
            return image
            
        except Exception as e:
            print(f"Ошибка создания демо-изображения: {e}")
            return None

def main():
    """Главная функция запуска приложения."""
    
    # Проверка наличия критических зависимостей
    if missing_deps:
        st.error("⚠️ Обнаружены проблемы с зависимостями")
        
        st.markdown("### 🛠️ Инструкции по устранению")
        
        st.markdown("""
        **Шаг 1: Обновите pip и установите зависимости**
        ```bash
        pip install --upgrade pip
        pip install -r requirements.txt
        ```
        
        **Шаг 2: Если OpenCV не устанавливается, попробуйте:**
        ```bash
        pip install opencv-python-headless==4.5.5.64
        ```
        
        **Шаг 3: Для Ubuntu/Debian также установите:**
        ```bash
        sudo apt-get update
        sudo apt-get install python3-opencv
        ```
        
        **Шаг 4: Перезапустите приложение:**
        ```bash
        streamlit run main.py
        ```
        """)
        
        if st.button("🔄 Перезагрузить приложение"):
            st.rerun()
        
        # Показываем детали ошибок
        with st.expander("🔍 Детали ошибок"):
            for dep in missing_deps:
                st.code(dep)
        
        return
    
    # Если все зависимости на месте, запускаем приложение
    app = CreativePerformanceApp()
    app.run()

if __name__ == "__main__":
    main()
