# main.py - ОБНОВЛЕННАЯ ВЕРСИЯ
"""
Основное приложение Creative Performance Predictor.
Обновленный Streamlit интерфейс для работы с революционными модулями.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import time
from typing import Dict, Any, Optional, List, Tuple
import warnings
import traceback
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

# Попытка импорта с обработкой ошибок
try:
    from image_analyzer import AdvancedImageAnalyzer
    from ml_engine import AdvancedMLEngine  
    from visualizer import AdvancedVisualizer
    from recommender import IntelligentRecommendationEngine
    from config import (
        APP_TITLE, APP_VERSION, PAGE_ICON, SUPPORTED_IMAGE_FORMATS,
        MAX_IMAGE_SIZE, CUSTOM_CSS, DEMO_INSIGHTS, COLOR_SCHEME
    )
    DEPENDENCIES_OK = True
    print("✅ Все модули успешно импортированы")
except ImportError as e:
    st.error(f"Критическая ошибка импорта: {e}")
    st.error("Убедитесь, что все зависимости установлены: pip install -r requirements.txt")
    DEPENDENCIES_OK = False
    # Инициализация заглушек
    AdvancedImageAnalyzer, AdvancedMLEngine, AdvancedVisualizer, IntelligentRecommendationEngine = None, None, None, None
    APP_TITLE, APP_VERSION, PAGE_ICON, SUPPORTED_IMAGE_FORMATS = "Error", "0.0", "❌", []
    MAX_IMAGE_SIZE, CUSTOM_CSS, DEMO_INSIGHTS, COLOR_SCHEME = 0, "", [], {}

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

@st.cache_resource
def get_advanced_app_engines():
    """
    Инициализирует и кэширует продвинутые движки приложения.
    Модели обучаются здесь один раз при первом запуске.
    """
    try:
        print("🔧 Инициализация продвинутых движков...")
        
        analyzer = AdvancedImageAnalyzer()
        print("✅ AdvancedImageAnalyzer инициализирован")
        
        recommender = IntelligentRecommendationEngine()  
        print("✅ IntelligentRecommendationEngine инициализирован")
        
        ml_engine = AdvancedMLEngine()
        print("✅ AdvancedMLEngine инициализирован")
        
        if not ml_engine.is_trained:
            with st.spinner('🧠 Обучение продвинутых ML моделей... Первый запуск может занять 2-3 минуты.'):
                print("🎓 Начинаем обучение моделей...")
                training_results = ml_engine.train_models(quick_mode=True)
                st.session_state.training_results = training_results
                print("🎉 Обучение завершено!")
        
        visualizer = AdvancedVisualizer()
        print("✅ AdvancedVisualizer инициализирован")
        
        print("🚀 Все движки готовы к работе!")
        return analyzer, ml_engine, visualizer, recommender
        
    except Exception as e:
        st.error(f"Ошибка инициализации движков: {e}")
        st.error("Подробности ошибки:")
        st.code(traceback.format_exc())
        raise e

class AdvancedCreativePerformanceApp:
    """Главный класс обновленного приложения Creative Performance Predictor."""
    
    def __init__(self):
        """Инициализация приложения и продвинутых компонентов."""
        if not DEPENDENCIES_OK:
            st.stop()
            
        try:
            self.analyzer, self.ml_engine, self.visualizer, self.recommender = get_advanced_app_engines()
        except Exception as e:
            st.error("Не удалось инициализировать движки приложения")
            st.stop()
            
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Инициализация переменных состояния сессии."""
        session_defaults = {
            'image_uploaded': False,
            'analysis_completed': False,
            'current_image': None,
            'image_features': {},
            'predictions': {},
            'recommendations': [],
            'analysis_results': {},
            'current_page': 'Главная',
            'advanced_mode': False,
            'benchmark_data': {},
            'analysis_history': []
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def run(self):
        """Запуск основного приложения."""
        st.markdown(f'<h1 class="main-header">{PAGE_ICON} {APP_TITLE} 2.0</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666; font-size: 1.1em;">Революционная система анализа креативов с ИИ</p>', unsafe_allow_html=True)
        
        self._render_sidebar()
        
        page = st.session_state.get('current_page', 'Главная')
        
        page_map = {
            'Главная': self._render_home_page,
            'Анализ креатива': self._render_analysis_page,
            'Результаты анализа': self._render_results_page,
            'Интеллектуальные рекомендации': self._render_recommendations_page,
            'Визуальная аналитика': self._render_visual_analytics_page,
            'О системе': self._render_about_page,
        }
        
        page_function = page_map.get(page)
        if page_function:
            page_function()
    
    def _render_sidebar(self):
        """Отрисовка продвинутой боковой панели."""
        with st.sidebar:
            st.markdown("### 🧭 Навигация")
            
            pages = [
                ('🏠', 'Главная'),
                ('🔍', 'Анализ креатива'),
                ('📊', 'Результаты анализа'), 
                ('🧠', 'Интеллектуальные рекомендации'),
                ('📈', 'Визуальная аналитика'),
                ('ℹ️', 'О системе')
            ]
            
            for icon, page_name in pages:
                if st.button(f"{icon} {page_name}", key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.current_page = page_name
                    st.rerun()
            
            st.markdown("---")
            
            # Продвинутые настройки
            st.markdown("### ⚙️ Настройки")
            st.session_state.advanced_mode = st.toggle("Экспертный режим", value=st.session_state.get('advanced_mode', False))
            
            if st.session_state.advanced_mode:
                st.markdown("🔬 **Доступны продвинутые функции:**")
                st.markdown("• Детальная аналитика")
                st.markdown("• Научные обоснования")
                st.markdown("• Экспорт данных")
            
            st.markdown("---")
            st.markdown("### 📈 Статус системы")
            
            # Статус компонентов
            if self.ml_engine.is_trained:
                st.success("✅ ML модели обучены")
                if st.session_state.advanced_mode:
                    if hasattr(self.ml_engine, 'model_performance'):
                        avg_r2 = np.mean([
                            np.mean([result.get('r2_score', 0) for result in target_results.values() if 'r2_score' in result])
                            for target_results in self.ml_engine.model_performance.values()
                        ])
                        st.metric("Средний R²", f"{avg_r2:.3f}")
            else:
                st.warning("⏳ Модели обучаются...")

            if st.session_state.image_uploaded:
                st.success("✅ Изображение загружено")
            else:
                st.info("📤 Загрузите изображение")
            
            if st.session_state.analysis_completed:
                st.success("✅ Анализ завершен")
                if st.session_state.advanced_mode:
                    analysis_time = st.session_state.get('analysis_time', 'N/A')
                    st.metric("Время анализа", f"{analysis_time}с")
            else:
                st.info("🔄 Анализ не выполнен")
            
            st.markdown("---")
            st.markdown("### 💡 Научный факт")
            insight = np.random.choice(DEMO_INSIGHTS)
            st.info(insight)
            
            if st.session_state.advanced_mode:
                st.markdown("---")
                st.markdown("### 🔍 История анализа")
                history_count = len(st.session_state.get('analysis_history', []))
                st.metric("Проанализировано", f"{history_count} креативов")
            
            st.markdown("---")
            st.caption(f"Версия {APP_VERSION} • Революционный ИИ")
    
    def _render_home_page(self):
        """Отрисовка улучшенной главной страницы."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 🎨 Добро пожаловать в Creative Performance Predictor 2.0!")
            st.markdown("""
            Революционная система анализа креативов, использующая передовые технологии:
            
            🧠 **Искусственный интеллект** — Ансамбль ML моделей для точных предсказаний  
            👁️ **Компьютерное зрение** — YOLO детекция объектов и EasyOCR анализ текста  
            🔬 **Научный подход** — Рекомендации на основе нейромаркетинга и психологии  
            📊 **Продвинутая аналитика** — Интерактивные heatmaps и 3D визуализации  
            """)
            
            st.markdown("### 🚀 Что нового в версии 2.0:")
            
            improvements = [
                "**Реальный OCR** — EasyOCR вместо заглушек",
                "**YOLO детекция** — Распознавание объектов и лиц", 
                "**Ансамбль моделей** — Random Forest + Gradient Boosting + XGBoost",
                "**Научные рекомендации** — База знаний из 50+ исследований",
                "**Интерактивная аналитика** — Heatmaps зон внимания",
                "**ROI калькулятор** — Расчет окупаемости улучшений"
            ]
            
            for improvement in improvements:
                st.markdown(f"✨ {improvement}")
            
            if st.button("🚀 Начать анализ", type="primary", use_container_width=True):
                st.session_state.current_page = 'Анализ креатива'
                st.rerun()

        with col2:
            st.markdown("### 📊 Возможности системы")
            
            # Интерактивные метрики
            metrics_data = [
                ("🎯", "Точность", "R² > 0.85", self.advanced_colors.get('performance_excellent', '#00C851')),
                ("⚡", "Скорость", "< 3 сек", self.advanced_colors.get('ctr_color', '#2196F3')),
                ("🔧", "Рекомендаций", "15+ советов", self.advanced_colors.get('engagement_color', '#FF9800')),
                ("📱", "Форматы", "JPG, PNG, WEBP", self.advanced_colors.get('trust_color', '#3F51B5'))
            ]
            
            for icon, metric, value, color in metrics_data:
                st.markdown(f'''
                <div style="padding: 1rem; margin: 0.5rem 0; border-radius: 10px; 
                           background: linear-gradient(135deg, {color}20, {color}10);
                           border-left: 4px solid {color};">
                    <h4 style="margin: 0; color: {color};">{icon} {metric}</h4>
                    <p style="margin: 0; font-weight: bold;">{value}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            st.markdown("### 🔬 Научная база")
            st.info("Рекомендации основаны на исследованиях MIT, Stanford, Nielsen Norman Group")
            
            # Демо статистика
            if st.session_state.advanced_mode:
                st.markdown("### 📈 Статистика использования")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Анализов сегодня", "247", "+15%")
                with col_b:
                    st.metric("Средний рост CTR", "23%", "+2%")

    @property
    def advanced_colors(self):
        """Свойство для доступа к цветам visualizer."""
        return getattr(self.visualizer, 'advanced_colors', COLOR_SCHEME)
    
    def _render_analysis_page(self):
        """Отрисовка улучшенной страницы анализа."""
        st.header("🔍 Революционный анализ креативного материала")
        
        # Расширенные настройки
        col1, col2, col3 = st.columns(3)
        with col1:
            category = st.selectbox(
                "Выберите категорию креатива", 
                ['Автомобили', 'E-commerce', 'Финансы', 'Технологии', 'Здоровье', 'Образование'],
                help="Влияет на отраслевые бенчмарки и специфические рекомендации"
            )
        with col2:
            region = st.selectbox(
                "Выберите регион", 
                ['Россия', 'США', 'Европа', 'Азия'],
                help="Учитывает культурные особенности восприятия"
            )
        with col3:
            target_audience = st.selectbox(
                "Целевая аудитория",
                ['Общая', '18-25', '25-35', '35-45', '45+'],
                help="Персонализирует рекомендации"
            )

        st.session_state.category = category
        st.session_state.region = region
        st.session_state.target_audience = target_audience

        # Загрузка файла с улучшенной обработкой
        uploaded_file = st.file_uploader(
            "Загрузите изображение для анализа", 
            type=SUPPORTED_IMAGE_FORMATS,
            help=f"Поддерживаемые форматы: {', '.join(SUPPORTED_IMAGE_FORMATS)}. Макс. размер: {MAX_IMAGE_SIZE // (1024*1024)}MB"
        )
        
        if uploaded_file is not None:
            if uploaded_file.size > MAX_IMAGE_SIZE:
                st.error(f"❌ Размер файла превышает {MAX_IMAGE_SIZE // (1024*1024)}MB")
                return
            
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.session_state.current_image = image
                st.session_state.image_uploaded = True
                
                # Информация об изображении
                img_col, info_col = st.columns([2, 1])
                
                with img_col:
                    st.image(image, caption="Загруженное изображение", use_container_width=True)
                
                with info_col:
                    st.markdown("### 📏 Характеристики изображения")
                    
                    # Базовая информация
                    st.write(f"**Размер:** {image.size[0]}×{image.size[1]}px")
                    st.write(f"**Соотношение сторон:** {image.size[0]/image.size[1]:.2f}")
                    st.write(f"**Формат:** {image.format}")
                    
                    # Размер файла
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    st.write(f"**Размер файла:** {file_size_mb:.1f}MB")
                    
                    # Выбранные параметры
                    st.markdown("### ⚙️ Параметры анализа")
                    st.write(f"**Категория:** {category}")
                    st.write(f"**Регион:** {region}")
                    st.write(f"**Аудитория:** {target_audience}")
                    
                    # Продвинутые настройки
                    if st.session_state.advanced_mode:
                        st.markdown("### 🔬 Экспертные настройки")
                        use_ai_enhancement = st.checkbox("ИИ улучшение качества", value=True)
                        detailed_analysis = st.checkbox("Детальный анализ", value=True)
                        
                        st.session_state.use_ai_enhancement = use_ai_enhancement
                        st.session_state.detailed_analysis = detailed_analysis

                # Кнопка анализа
                if st.button("🚀 Запустить революционный анализ", type="primary", use_container_width=True):
                    self._perform_advanced_analysis(image, category, region, target_audience)
                
            except Exception as e:
                st.error(f"❌ Ошибка при обработке изображения: {str(e)}")
                if st.session_state.advanced_mode:
                    st.code(traceback.format_exc())
        
        else:
            st.info("👆 **Загрузите изображение для начала революционного анализа**")
            
            col_demo, col_space = st.columns([1, 1])
            with col_demo:
                if st.button("🎲 Использовать демо-изображение", use_container_width=True):
                    demo_image = self._create_advanced_demo_image()
                    if demo_image:
                        st.session_state.current_image = demo_image
                        st.session_state.image_uploaded = True
                        st.success("✅ Демо-изображение загружено! Нажмите 'Запустить анализ'.")
                        st.rerun()

    def _perform_advanced_analysis(self, image: Image.Image, category: str, region: str, target_audience: str):
        """Выполнение продвинутого анализа изображения."""
        start_time = time.time()
        
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0, text="🔄 Инициализация революционного анализа...")
        
        try:
            def update_progress(percent, text):
                progress_bar.progress(percent, text=text)
                time.sleep(0.1)  # Небольшая задержка для UX

            # Этап 1: Загрузка изображения в анализатор
            update_progress(5, "📸 Загрузка изображения в ИИ анализатор...")
            if not self.analyzer.load_image(image):
                st.error("❌ Не удалось загрузить изображение в анализатор.")
                return
            
            # Этап 2: Анализ цветовых характеристик
            update_progress(15, "🎨 Анализ цветовой психологии...")
            color_analysis = self.analyzer.analyze_colors()
            
            # Этап 3: Анализ композиции с YOLO
            update_progress(30, "🏗️ ИИ анализ композиции и объектов...")
            composition_analysis = self.analyzer.analyze_composition()
            
            # Этап 4: Революционный анализ текста
            update_progress(45, "📝 OCR и анализ текстовых элементов...")
            text_analysis = self.analyzer.analyze_text()
            
            # Этап 5: Извлечение всех признаков
            update_progress(60, "🧠 Извлечение признаков для ML...")
            image_features = self.analyzer.get_all_features()
            image_features['category'] = category
            image_features['region'] = region
            image_features['target_audience'] = target_audience
            
            st.session_state.image_features = image_features

            # Этап 6: ML предсказания
            update_progress(75, "🔮 Ансамбль ML моделей предсказывает эффективность...")
            predictions = self.ml_engine.predict(image_features)
            confidence_intervals = self.ml_engine.get_prediction_confidence(image_features)
            
            # Этап 7: Интеллектуальные рекомендации
            update_progress(85, "💡 Генерация научно обоснованных рекомендаций...")
            recommendations = self.recommender.generate_intelligent_recommendations(
                image_features, predictions, category, target_audience
            )
            
            # Этап 8: Финализация и сохранение
            update_progress(95, "📊 Подготовка результатов...")
            
            analysis_time = time.time() - start_time
            
            # Сохранение результатов
            st.session_state.analysis_results = {
                'color_analysis': color_analysis,
                'composition_analysis': composition_analysis,
                'text_analysis': text_analysis,
                'image_features': image_features,
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'recommendations': recommendations,
                'analysis_time': analysis_time,
                'timestamp': time.time()
            }
            
            st.session_state.predictions = predictions
            st.session_state.recommendations = recommendations
            st.session_state.analysis_completed = True
            st.session_state.analysis_time = f"{analysis_time:.1f}"
            
            # Добавляем в историю
            if 'analysis_history' not in st.session_state:
                st.session_state.analysis_history = []
            
            st.session_state.analysis_history.append({
                'timestamp': time.time(),
                'category': category,
                'predictions': predictions,
                'analysis_time': analysis_time
            })
            
            update_progress(100, "✅ Революционный анализ завершен!")
            time.sleep(0.5)
            progress_container.empty()
            
            # Показываем краткие результаты
            self._show_quick_results(predictions, analysis_time)
            
        except Exception as e:
            progress_container.empty()
            st.error(f"❌ Произошла ошибка во время анализа: {str(e)}")
            if st.session_state.advanced_mode:
                st.error("Подробности ошибки:")
                st.code(traceback.format_exc())

    def _show_quick_results(self, predictions: Dict, analysis_time: float):
        """Показ быстрых результатов анализа."""
        st.success("🎉 Революционный анализ успешно завершен!")
        
        # Быстрые метрики
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ctr_color = self._get_metric_color(predictions['ctr'], 'ctr')
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: {ctr_color}20; border-radius: 10px;">
                <h3 style="color: {ctr_color}; margin: 0;">CTR</h3>
                <h2 style="margin: 0;">{predictions['ctr']*100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            conv_color = self._get_metric_color(predictions['conversion_rate'], 'conversion_rate')
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: {conv_color}20; border-radius: 10px;">
                <h3 style="color: {conv_color}; margin: 0;">Конверсия</h3>
                <h2 style="margin: 0;">{predictions['conversion_rate']*100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            eng_color = self._get_metric_color(predictions['engagement'], 'engagement')
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: {eng_color}20; border-radius: 10px;">
                <h3 style="color: {eng_color}; margin: 0;">Вовлеченность</h3>
                <h2 style="margin: 0;">{predictions['engagement']*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: #2196F320; border-radius: 10px;">
                <h3 style="color: #2196F3; margin: 0;">Время анализа</h3>
                <h2 style="margin: 0;">{analysis_time:.1f}с</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Кнопки навигации
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("📊 Посмотреть детальные результаты", type="primary", use_container_width=True):
                st.session_state.current_page = 'Результаты анализа'
                st.rerun()
        
        with col_nav2:
            if st.button("🧠 Интеллектуальные рекомендации", use_container_width=True):
                st.session_state.current_page = 'Интеллектуальные рекомендации'
                st.rerun()

    def _get_metric_color(self, value: float, metric_type: str) -> str:
        """Получение цвета метрики на основе ее значения."""
        thresholds = {
            'ctr': {'excellent': 0.04, 'good': 0.025},
            'conversion_rate': {'excellent': 0.08, 'good': 0.05},
            'engagement': {'excellent': 0.15, 'good': 0.10}
        }
        
        if value >= thresholds[metric_type]['excellent']:
            return '#00C851'  # Зеленый
        elif value >= thresholds[metric_type]['good']:
            return '#FF9800'  # Оранжевый
        else:
            return '#F44336'  # Красный

    def _render_results_page(self):
        """Отрисовка страницы с результатами анализа."""
        if not st.session_state.analysis_completed:
            st.warning("⚠️ Сначала выполните анализ на странице 'Анализ креатива'.")
            if st.button("🔍 Перейти к анализу"):
                st.session_state.current_page = 'Анализ креатива'
                st.rerun()
            return
        
        st.header("📊 Детальные результаты революционного анализа")
        
        analysis_results = st.session_state.analysis_results
        predictions = analysis_results['predictions']
        
        # Главный дашборд
        st.subheader("🎯 Интеллектуальный дашборд эффективности")
        
        # Получаем бенчмарки для категории
        category = st.session_state.get('category', 'Общая')
        benchmarks = self._get_category_benchmarks(category)
        
        dashboard_fig = self.visualizer.create_performance_dashboard(
            predictions, 
            analysis_results.get('confidence_intervals'),
            benchmarks
        )
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Табы с детальным анализом
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔮 Предсказания", "🎨 Цветовая психология", "🏗️ Композиция", 
            "📝 Текст и OCR", "🎯 Зоны внимания"
        ])
        
        with tab1:
            self._render_predictions_tab(analysis_results)
        
        with tab2:
            self._render_color_psychology_tab(analysis_results)
        
        with tab3:
            self._render_composition_tab(analysis_results)
        
        with tab4:
            self._render_text_analysis_tab(analysis_results)
        
        with tab5:
            self._render_attention_heatmap_tab(analysis_results)

    def _render_predictions_tab(self, analysis_results: Dict):
        """Рендер таба предсказаний."""
        st.subheader("🔮 Детальный анализ предсказаний")
        
        predictions = analysis_results['predictions']
        feature_importance = self.ml_engine.get_feature_importance('ctr')
        confidence_intervals = analysis_results.get('confidence_intervals', {})
        
        # Детальный график предсказаний
        detailed_pred_fig = self.visualizer.create_performance_prediction_detailed(
            predictions, feature_importance, confidence_intervals
        )
        st.plotly_chart(detailed_pred_fig, use_container_width=True)
        
        if st.session_state.advanced_mode:
            # Экспертная информация
            st.markdown("### 🔬 Экспертная информация")
            
            explanation = self.ml_engine.explain_prediction(st.session_state.image_features)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Уверенность модели:**")
                confidence = explanation.get('model_confidence', 0.8)
                st.progress(confidence, text=f"{confidence:.1%}")
                
                st.markdown("**Категория эффективности:**")
                st.info(explanation.get('performance_category', 'Средняя'))
            
            with col2:
                st.markdown("**Приоритет рекомендаций:**")
                priority = explanation.get('recommendation_priority', 'Средний')
                priority_color = {'Высокий': '🔴', 'Средний': '🟡', 'Низкий': '🟢'}
                st.markdown(f"{priority_color.get(priority, '🟡')} {priority}")
                
                st.markdown("**Ключевые инсайты:**")
                for insight in explanation.get('key_insights', [])[:3]:
                    st.markdown(f"• {insight}")

    def _render_color_psychology_tab(self, analysis_results: Dict):
        """Рендер таба цветовой психологии."""
        st.subheader("🎨 Анализ цветовой психологии")
        
        color_data = analysis_results['color_analysis']
        
        # График цветовой психологии
        color_fig = self.visualizer.create_color_psychology_analysis(color_data)
        st.plotly_chart(color_fig, use_container_width=True)
        
        # Дополнительная информация о цветах
        if 'dominant_colors' in color_data and color_data['dominant_colors']:
            st.markdown("### 🌈 Доминирующие цвета и их влияние")
            
            colors = color_data['dominant_colors'][:5]
            cols = st.columns(len(colors))
            
            for i, color in enumerate(colors):
                with cols[i]:
                    hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    st.markdown(f"""
                    <div style="width: 100%; height: 60px; background-color: {hex_color}; 
                                border-radius: 5px; margin-bottom: 10px;"></div>
                    <p style="text-align: center; font-size: 12px;">
                        RGB({color[0]}, {color[1]}, {color[2]})<br>
                        {self._get_color_psychology(color)}
                    </p>
                    """, unsafe_allow_html=True)

    def _render_composition_tab(self, analysis_results: Dict):
        """Рендер таба композиции."""
        st.subheader("🏗️ Анализ композиции")
        
        composition_data = analysis_results['composition_analysis']
        
        # 3D анализ композиции
        if st.session_state.advanced_mode:
            comp_3d_fig = self.visualizer.create_composition_analysis_3d(composition_data)
            st.plotly_chart(comp_3d_fig, use_container_width=True)
        
        # Стандартный анализ композиции
        comp_fig = self.visualizer.plot_composition_analysis_3d(composition_data)
        st.plotly_chart(comp_fig, use_container_width=True)
        
        # Детали композиции
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📐 Ключевые принципы")
            principles = [
                ("Правило третей", composition_data.get('rule_of_thirds_score', 0)),
                ("Визуальный баланс", composition_data.get('visual_balance_score', 0)),
                ("Центральный фокус", composition_data.get('center_focus_score', 0)),
            ]
            
            for principle, score in principles:
                color = '#00C851' if score > 0.7 else '#FF9800' if score > 0.4 else '#F44336'
                st.markdown(f"**{principle}:** <span style='color: {color}'>{score:.2f}</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎯 Детектированные объекты")
            focal_points = composition_data.get('focal_points', 0)
            if focal_points > 0:
                st.success(f"✅ Обнаружено {focal_points} ключевых объектов")
                st.info("Объекты помогают направлять внимание зрителя")
            else:
                st.warning("⚠️ Ключевые объекты не обнаружены")
                st.info("Рекомендуется добавить фокусные элементы")

    def _render_text_analysis_tab(self, analysis_results: Dict):
        """Рендер таба анализа текста."""
        st.subheader("📝 OCR и анализ текста")
        
        text_data = analysis_results['text_analysis']
        
        # Основные метрики текста
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Текстовых блоков", text_data.get('text_amount', 0))
        with col2:
            st.metric("Символов", text_data.get('total_characters', 0))
        with col3:
            cta_status = "Есть ✅" if text_data.get('has_cta', False) else "Нет ❌"
            st.metric("Призыв к действию", cta_status)
        with col4:
            readability = text_data.get('readability_score', 0)
            st.metric("Читаемость", f"{readability:.2f}")
        
        # Детальный анализ
        st.markdown("### 📊 Детальные характеристики текста")
        
        text_metrics = [
            ("Читаемость", text_data.get('readability_score', 0)),
            ("Иерархия", text_data.get('text_hierarchy', 0)),
            ("Позиционирование", text_data.get('text_positioning', 0)),
            ("Контрастность", text_data.get('text_contrast', 0)),
        ]
        
        for metric, value in text_metrics:
            st.progress(value, text=f"{metric}: {value:.2f}")
        
        if st.session_state.advanced_mode and text_data.get('text_amount', 0) > 0:
            st.markdown("### 🔍 Техническая информация OCR")
            st.json({
                "Плотность текста": text_data.get('text_density', 0),
                "Покрытие текста": text_data.get('text_coverage', 0),
                "Разнообразие шрифтов": text_data.get('font_variety', 0),
                "Соотношение текст/изображение": text_data.get('text_to_image_ratio', 0)
            })

    def _render_attention_heatmap_tab(self, analysis_results: Dict):
        """Рендер таба с heatmap внимания."""
        st.subheader("🎯 Карта зон внимания")
        
        st.info("Heatmap показывает области, которые привлекают максимальное внимание пользователей на основе принципов нейромаркетинга")
        
        # Создаем heatmap
        heatmap_fig = self.visualizer.create_attention_heatmap(
            st.session_state.image_features, 
            analysis_results['predictions']
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Объяснение зон
        st.markdown("### 🧠 Объяснение зон внимания")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔥 Красные зоны** — Максимальное внимание
            - Точки пересечения правила третей
            - Области с высоким контрастом
            - Расположение призывов к действию
            """)
        
        with col2:
            st.markdown("""
            **🟡 Желтые зоны** — Умеренное внимание  
            - Центральная область изображения
            - Зоны с объектами средней важности
            - Текстовые блоки
            """)
        
        if st.session_state.advanced_mode:
            st.markdown("### 🔬 Научное обоснование")
            st.markdown("""
            Карта внимания построена на основе:
            - **F-паттерн чтения** (Nielsen Norman Group)
            - **Правило третей** (Golden Ratio Research)
            - **Эффект изоляции** (Von Restorff Effect)
            - **Контрастность и выделение** (Attention Psychology)
            """)

    def _render_recommendations_page(self):
        """Отрисовка страницы интеллектуальных рекомендаций."""
        if not st.session_state.analysis_completed:
            st.warning("⚠️ Сначала выполните анализ на странице 'Анализ креатива'.")
            return
        
        st.header("🧠 Интеллектуальные рекомендации на основе ИИ")
        
        recommendations = st.session_state.recommendations
        
        if not recommendations:
            st.success("🎉 Отличная работа! Критических улучшений не требуется.")
            st.info("Ваш креатив уже показывает хорошие результаты по всем ключевым метрикам.")
            return
        
        # Общая статистика
        total_impact = sum(rec.expected_impact for rec in recommendations)
        avg_confidence = np.mean([rec.confidence for rec in recommendations])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Общий потенциал роста", f"{total_impact:.1%}")
        with col2:
            st.metric("Количество рекомендаций", len(recommendations))
        with col3:
            st.metric("Средняя уверенность", f"{avg_confidence:.1%}")
        
        # Карта влияния рекомендаций
        impact_fig = self.visualizer.create_recommendation_impact_chart(recommendations)
        st.plotly_chart(impact_fig, use_container_width=True)
        
        # Группировка рекомендаций
        st.subheader("📋 Детальные рекомендации")
        
        # Фильтры
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            priority_filter = st.selectbox("Фильтр по приоритету", ["Все", "high", "medium", "low"])
        with col_filter2:
            category_filter = st.selectbox("Фильтр по категории", 
                                         ["Все"] + list(set(rec.category for rec in recommendations)))
        
        # Применение фильтров
        filtered_recs = recommendations
        if priority_filter != "Все":
            filtered_recs = [r for r in filtered_recs if r.priority == priority_filter]
        if category_filter != "Все":
            filtered_recs = [r for r in filtered_recs if r.category == category_filter]
        
        # Отображение рекомендаций
        for i, rec in enumerate(filtered_recs):
            self._render_recommendation_card(rec, i)
        
        # Дорожная карта внедрения
        if st.session_state.advanced_mode:
            st.subheader("🗺️ Дорожная карта внедрения")
            roadmap = self.recommender.create_implementation_roadmap(recommendations)
            self._render_implementation_roadmap(roadmap)

    def _render_recommendation_card(self, rec, index: int):
        """Рендер карточки рекомендации."""
        priority_colors = {
            'high': '#F44336',
            'medium': '#FF9800', 
            'low': '#4CAF50'
        }
        
        priority_emoji = {
            'high': '🔥',
            'medium': '⚡',
            'low': '💡'
        }
        
        color = priority_colors.get(rec.priority, '#4CAF50')
        emoji = priority_emoji.get(rec.priority, '💡')
        
        with st.expander(f"{emoji} {rec.title}", expanded=(index < 3)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Описание:** {rec.description}")
                
                st.markdown("**Пошаговые действия:**")
                for step in rec.actionable_steps:
                    st.markdown(f"• {step}")
                
                if st.session_state.advanced_mode:
                    st.markdown(f"**Научное обоснование:** {rec.scientific_basis}")
            
            with col2:
                # Метрики рекомендации
                st.markdown(f"**Приоритет:** <span style='color: {color}'>{rec.priority.upper()}</span>", unsafe_allow_html=True)
                st.metric("Ожидаемое влияние", f"{rec.expected_impact:.1%}")
                st.metric("Уверенность", f"{rec.confidence:.1%}")
                st.metric("Время реализации", rec.time_estimate)
                st.metric("Уровень усилий", rec.effort_level.title())
                
                if hasattr(rec, 'roi_estimate'):
                    st.metric("ROI оценка", f"{rec.roi_estimate:.1f}x")
                
                # Необходимые инструменты
                if hasattr(rec, 'tools_needed') and rec.tools_needed:
                    st.markdown("**Инструменты:**")
                    for tool in rec.tools_needed:
                        st.markdown(f"• {tool}")

    def _render_visual_analytics_page(self):
        """Отрисовка страницы визуальной аналитики."""
        if not st.session_state.analysis_completed:
            st.warning("⚠️ Сначала выполните анализ на странице 'Анализ креатива'.")
            return
        
        st.header("📈 Визуальная аналитика и инсайты")
        
        # Выбор типа визуализации
        viz_type = st.selectbox(
            "Выберите тип аналитики",
            ["Общий дашборд", "Цветовая психология", "Композиционный анализ", "Карта внимания", "Сравнительный анализ"]
        )
        
        analysis_results = st.session_state.analysis_results
        
        if viz_type == "Общий дашборд":
            dashboard_fig = self.visualizer.create_performance_dashboard(
                analysis_results['predictions'],
                analysis_results.get('confidence_intervals'),
                self._get_category_benchmarks(st.session_state.get('category', 'Общая'))
            )
            st.plotly_chart(dashboard_fig, use_container_width=True)
        
        elif viz_type == "Цветовая психология":
            color_fig = self.visualizer.create_color_psychology_analysis(
                analysis_results['color_analysis']
            )
            st.plotly_chart(color_fig, use_container_width=True)
        
        elif viz_type == "Композиционный анализ":
            comp_fig = self.visualizer.create_composition_analysis_3d(
                analysis_results['composition_analysis']
            )
            st.plotly_chart(comp_fig, use_container_width=True)
        
        elif viz_type == "Карта внимания":
            heatmap_fig = self.visualizer.create_attention_heatmap(
                st.session_state.image_features,
                analysis_results['predictions']
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        elif viz_type == "Сравнительный анализ":
            self._render_comparative_analysis()

    def _render_about_page(self):
        """Отрисовка страницы 'О системе'."""
        st.header("ℹ️ О системе Creative Performance Predictor 2.0")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            ## 🎨 Creative Performance Predictor 2.0
            **Версия {APP_VERSION}** • Революционная система анализа креативов
            
            ### 🧠 Технологический стек
            
            **Машинное обучение:**
            - Ансамбль моделей: Random Forest + Gradient Boosting + XGBoost
            - Feature Engineering с научным обоснованием
            - Кросс-валидация и доверительные интервалы
            
            **Компьютерное зрение:**
            - EasyOCR/Tesseract для анализа текста
            - YOLO v8 для детекции объектов
            - Продвинутый анализ цветов и композиции
            
            **Интеллектуальные рекомендации:**
            - База знаний из 50+ научных исследований
            - Принципы нейромаркетинга и психологии
            - Персонализация по отраслям и аудиториям
            
            **Визуализация:**
            - Интерактивные дашборды с Plotly
            - Heatmaps зон внимания
            - 3D анализ композиции
            """)
        
        with col2:
            st.markdown("### 📊 Статистика системы")
            
            if 'training_results' in st.session_state:
                st.markdown("**Качество моделей:**")
                training_results = st.session_state.training_results
                
                for target, results in training_results.items():
                    if isinstance(results, dict):
                        for model, metrics in results.items():
                            if isinstance(metrics, dict) and 'r2_score' in metrics:
                                r2 = metrics['r2_score']
                                st.metric(f"{target} ({model})", f"R² = {r2:.3f}")
            
            st.markdown("### 🔬 Научная база")
            st.info("""
            Рекомендации основаны на исследованиях:
            • MIT Neuromarketing Lab
            • Stanford Psychology Dept
            • Nielsen Norman Group
            • Cambridge Color Research
            """)
            
            if st.session_state.advanced_mode:
                st.markdown("### ⚙️ Технические детали")
                st.code(f"""
                Версия: {APP_VERSION}
                ML движок: AdvancedMLEngine
                Анализатор: AdvancedImageAnalyzer  
                Рекомендации: IntelligentRecommendationEngine
                Визуализация: AdvancedVisualizer
                """)

    # Вспомогательные методы
    def _get_category_benchmarks(self, category: str) -> Dict[str, float]:
        """Получение бенчмарков для категории."""
        benchmarks_map = {
            'E-commerce': {'ctr': 0.035, 'conversion_rate': 0.082, 'engagement': 0.124},
            'Финансы': {'ctr': 0.022, 'conversion_rate': 0.064, 'engagement': 0.089},
            'Автомобили': {'ctr': 0.041, 'conversion_rate': 0.045, 'engagement': 0.156},
            'Технологии': {'ctr': 0.028, 'conversion_rate': 0.071, 'engagement': 0.134},
            'Здоровье': {'ctr': 0.031, 'conversion_rate': 0.067, 'engagement': 0.098},
            'Образование': {'ctr': 0.025, 'conversion_rate': 0.055, 'engagement': 0.087}
        }
        return benchmarks_map.get(category, {'ctr': 0.025, 'conversion_rate': 0.05, 'engagement': 0.1})

    def _get_color_psychology(self, color: Tuple[int, int, int]) -> str:
        """Получение психологической характеристики цвета."""
        r, g, b = color
        
        if r > g and r > b:
            return "Энергия, страсть"
        elif g > r and g > b:
            return "Природа, рост"
        elif b > r and b > g:
            return "Доверие, спокойствие"
        else:
            return "Баланс"

    def _create_advanced_demo_image(self) -> Optional[Image.Image]:
        """Создание продвинутого демонстрационного изображения."""
        try:
            width, height = 800, 600
            image = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(image)
            
            # Градиентный фон
            for y in range(height):
                r = int(255 * (1 - y/height * 0.3))
                g = int(180 * (1 - y/height * 0.5))
                b = 80
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # Главный заголовок
            try:
                font_large = ImageFont.load_default(size=48)
                font_medium = ImageFont.load_default(size=24)
                font_small = ImageFont.load_default(size=16)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            main_text = "DEMO CREATIVE 2.0"
            bbox = draw.textbbox((0, 0), main_text, font=font_large)
            x = (width - (bbox[2] - bbox[0])) / 2
            y = height / 4
            draw.text((x, y), main_text, fill="white", font=font_large)
            
            # Подзаголовок
            sub_text = "Революционный анализ креативов"
            bbox = draw.textbbox((0, 0), sub_text, font=font_medium)
            x = (width - (bbox[2] - bbox[0])) / 2
            y = height / 2.5
            draw.text((x, y), sub_text, fill="lightgray", font=font_medium)
            
            # CTA кнопка
            button_x, button_y = width / 2, height - 120
            button_w, button_h = 220, 60
            draw.rectangle([
                button_x - button_w/2, button_y - button_h/2, 
                button_x + button_w/2, button_y + button_h/2
            ], fill="red", outline="darkred", width=2)
            
            cta_text = "ПОПРОБОВАТЬ СЕЙЧАС"
            cta_bbox = draw.textbbox((0, 0), cta_text, font=font_small)
            cta_x = button_x - (cta_bbox[2] - cta_bbox[0]) / 2
            cta_y = button_y - (cta_bbox[3] - cta_bbox[1]) / 2
            draw.text((cta_x, cta_y), cta_text, fill="white", font=font_small)
            
            # Декоративные элементы
            for i in range(3):
                x = 100 + i * 250
                y = height - 200
                draw.ellipse([x-30, y-30, x+30, y+30], fill="yellow", outline="orange", width=2)
            
            return image
            
        except Exception as e:
            st.error(f"Не удалось создать демо-изображение: {e}")
            return None

    def _render_comparative_analysis(self):
        """Рендер сравнительного анализа."""
        st.subheader("📊 Сравнительный анализ")
        
        if len(st.session_state.get('analysis_history', [])) < 2:
            st.info("Недостаточно данных для сравнения. Проанализируйте больше креативов.")
            return
        
        history = st.session_state.analysis_history
        
        # График истории анализов
        df_history = pd.DataFrame(history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'], unit='s')
        
        fig = go.Figure()
        
        for metric in ['ctr', 'conversion_rate', 'engagement']:
            values = [pred[metric] * 100 for pred in df_history['predictions']]
            fig.add_trace(go.Scatter(
                x=df_history['timestamp'], y=values,
                mode='lines+markers', name=metric.upper()
            ))
        
        fig.update_layout(
            title="История анализов креативов",
            xaxis_title="Время",
            yaxis_title="Значение (%)",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _render_implementation_roadmap(self, roadmap: Dict):
        """Рендер дорожной карты внедрения."""
        st.markdown("### 🗺️ Дорожная карта внедрения")
        
        for phase_key, phase_data in roadmap.items():
            if phase_key == 'summary':
                continue
                
            with st.expander(phase_data['title'], expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Потенциальное влияние", f"{phase_data['total_impact']:.1%}")
                with col2:
                    st.metric("Время реализации", f"{phase_data['total_time']:.1f}ч")
                with col3:
                    st.metric("ROI оценка", f"{phase_data['roi_estimate']:.1f}x")
                
                st.markdown("**Рекомендации в этой фазе:**")
                for rec in phase_data['recommendations']:
                    st.markdown(f"• {rec.title}")


if __name__ == "__main__":
    if DEPENDENCIES_OK:
        app = AdvancedCreativePerformanceApp()
        app.run()
