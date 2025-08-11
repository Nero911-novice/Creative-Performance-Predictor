# main.py
"""
Основное приложение Creative Performance Predictor.
Streamlit интерфейс для анализа и оптимизации креативов.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import time
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта с обработкой ошибок
try:
    from image_analyzer import ImageAnalyzer
    from ml_engine import MLEngine
    from visualizer import Visualizer
    from recommender import RecommendationEngine
    from config import (
        APP_TITLE, APP_VERSION, PAGE_ICON, SUPPORTED_IMAGE_FORMATS,
        MAX_IMAGE_SIZE, CUSTOM_CSS, DEMO_INSIGHTS, COLOR_SCHEME
    )
    DEPENDENCIES_OK = True
except ImportError as e:
    st.error(f"Критическая ошибка импорта: {e}. Пожалуйста, установите все зависимости из requirements.txt.")
    DEPENDENCIES_OK = False
    # Инициализация заглушек, чтобы избежать NameError
    ImageAnalyzer, MLEngine, Visualizer, RecommendationEngine = None, None, None, None
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
def get_app_engines():
    """
    Инициализирует и кэширует основные движки приложения.
    Модель обучается здесь один раз при первом запуске.
    """
    analyzer = ImageAnalyzer()
    recommender = RecommendationEngine()
    ml_engine = MLEngine()

    if not ml_engine.is_trained:
        with st.spinner('🤖 Первичная настройка и обучение модели... Это может занять минуту.'):
            training_results = ml_engine.train_models(quick_mode=True)
            st.session_state.training_results = training_results
    
    visualizer = Visualizer()
    return analyzer, ml_engine, visualizer, recommender


class CreativePerformanceApp:
    """Главный класс приложения Creative Performance Predictor."""
    
    def __init__(self):
        """Инициализация приложения и компонентов."""
        if not DEPENDENCIES_OK:
            st.stop()
            
        self.analyzer, self.ml_engine, self.visualizer, self.recommender = get_app_engines()
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
            'current_page': 'Главная'
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def run(self):
        """Запуск основного приложения."""
        st.markdown(f'<h1 class="main-header">{PAGE_ICON} {APP_TITLE}</h1>', unsafe_allow_html=True)
        self._render_sidebar()
        
        page = st.session_state.get('current_page', 'Главная')
        
        page_map = {
            'Главная': self._render_home_page,
            'Анализ изображения': self._render_analysis_page,
            'Результаты': self._render_results_page,
            'Рекомендации': self._render_recommendations_page,
            'О проекте': self._render_about_page,
        }
        
        page_function = page_map.get(page)
        if page_function:
            page_function()
    
    def _render_sidebar(self):
        """Отрисовка боковой панели."""
        with st.sidebar:
            st.markdown("### 🧭 Навигация")
            
            pages = [
                ('🏠', 'Главная'),
                ('🔍', 'Анализ изображения'),
                ('📊', 'Результаты'),
                ('💡', 'Рекомендации'),
                ('ℹ️', 'О проекте')
            ]
            
            for icon, page_name in pages:
                if st.button(f"{icon} {page_name}", key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.current_page = page_name
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### 📈 Статус системы")
            
            if self.ml_engine.is_trained:
                st.success("✅ Модель готова к работе")
            else:
                st.warning("⏳ Модель обучается...")

            if st.session_state.image_uploaded:
                st.success("✅ Изображение загружено")
            else:
                st.info("📤 Загрузите изображение")
            
            if st.session_state.analysis_completed:
                st.success("✅ Анализ завершен")
            else:
                st.info("🔄 Анализ не выполнен")
            
            st.markdown("---")
            st.markdown("### 💭 Знаете ли вы?")
            insight = np.random.choice(DEMO_INSIGHTS)
            st.info(insight)
            st.markdown("---")
            st.caption(f"Версия {APP_VERSION}")
    
    def _render_home_page(self):
        """Отрисовка главной страницы."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 🎨 Добро пожаловать в Creative Performance Predictor!")
            st.markdown("Это интеллектуальная система для анализа и оптимизации креативных материалов с использованием компьютерного зрения и машинного обучения.")
            st.markdown("### 🔍 Что умеет система:")
            st.markdown("- **Анализ изображений**: извлечение цветовых, композиционных и текстовых характеристик.\n"
                        "- **Предсказание эффективности**: прогноз CTR, конверсий и вовлеченности.\n"
                        "- **Персонализированные рекомендации**: конкретные советы по улучшению креативов.")
            
            if st.button("🚀 Начать анализ", type="primary"):
                st.session_state.current_page = 'Анализ изображения'
                st.rerun()

        with col2:
            st.markdown("### 📊 Возможности системы")
            metrics_data = [
                ("🎯", "Точность", "R² > 0.80"),
                ("⚡", "Скорость", "< 5 сек"),
                ("🔧", "Рекомендации", "10+ советов"),
                ("📱", "Форматы", "JPG, PNG, WEBP")
            ]
            for icon, metric, value in metrics_data:
                st.markdown(f'<div class="metric-card"><h4>{icon} {metric}</h4><p>{value}</p></div>', unsafe_allow_html=True)

    def _render_analysis_page(self):
        """Отрисовка страницы анализа изображения."""
        st.header("🔍 Анализ креативного материала")
        
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Выберите категорию креатива", ['Автомобили', 'E-commerce', 'Финансы', 'Технологии'])
        with col2:
            region = st.selectbox("Выберите регион", ['Россия', 'США', 'Европа'])

        st.session_state.category = category
        st.session_state.region = region

        uploaded_file = st.file_uploader("Загрузите изображение для анализа", type=SUPPORTED_IMAGE_FORMATS, help=f"Поддерживаемые форматы: {', '.join(SUPPORTED_IMAGE_FORMATS)}. Макс. размер: {MAX_IMAGE_SIZE // (1024*1024)}MB")
        
        if uploaded_file is not None:
            if uploaded_file.size > MAX_IMAGE_SIZE:
                st.error(f"Размер файла превышает {MAX_IMAGE_SIZE // (1024*1024)}MB")
                return
            
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.session_state.current_image = image
                st.session_state.image_uploaded = True
                
                img_col, info_col = st.columns([2, 1])
                with img_col:
                    st.image(image, caption="Загруженное изображение", use_container_width=True)
                with info_col:
                    st.markdown("### 📏 Характеристики")
                    st.write(f"**Размер:** {image.size[0]}×{image.size[1]}px")
                    st.write(f"**Формат:** {image.format}")
                    st.write(f"**Выбранная категория:** {category}")
                    st.write(f"**Выбранный регион:** {region}")

                if st.button("🚀 Запустить анализ", type="primary"):
                    self._perform_analysis(image, category, region)
                
            except Exception as e:
                st.error(f"Ошибка при обработке изображения: {str(e)}")
        
        else:
            st.info("👆 **Загрузите изображение, выберите категорию/регион и начните анализ**")
            if st.button("🎲 Использовать демо-изображение"):
                demo_image = self._create_demo_image()
                if demo_image:
                    st.session_state.current_image = demo_image
                    st.session_state.image_uploaded = True
                    st.success("✅ Демо-изображение загружено! Нажмите 'Запустить анализ'.")
                    st.rerun()

    def _perform_analysis(self, image: Image.Image, category: str, region: str):
        """Выполнение полного анализа изображения."""
        progress_bar = st.progress(0, text="🔄 Подготовка к анализу...")
        
        try:
            def update_progress(percent, text):
                progress_bar.progress(percent, text=text)
                time.sleep(0.2)

            update_progress(10, "🎨 Анализ цветовых характеристик...")
            if not self.analyzer.load_image(image):
                st.error("Не удалось загрузить изображение в анализатор.")
                return
            
            color_analysis = self.analyzer.analyze_colors()
            update_progress(30, "📐 Анализ композиции...")
            composition_analysis = self.analyzer.analyze_composition()
            update_progress(50, "📝 Анализ текстовых элементов...")
            text_analysis = self.analyzer.analyze_text()
            update_progress(70, "🧠 Подготовка данных для ML...")
            image_features = self.analyzer.get_all_features()
            image_features['category'] = category
            image_features['region'] = region
            st.session_state.image_features = image_features

            update_progress(80, "🔮 Предсказание эффективности...")
            predictions = self.ml_engine.predict(image_features)
            confidence_intervals = self.ml_engine.get_prediction_confidence(image_features)
            update_progress(90, "💡 Генерация рекомендаций...")
            recommendations = self.recommender.generate_recommendations(image_features, predictions)
            
            st.session_state.analysis_results = {
                'color_analysis': color_analysis, 'composition_analysis': composition_analysis,
                'text_analysis': text_analysis, 'image_features': image_features,
                'predictions': predictions, 'confidence_intervals': confidence_intervals,
                'recommendations': recommendations
            }
            st.session_state.predictions = predictions
            st.session_state.recommendations = recommendations
            st.session_state.analysis_completed = True
            
            update_progress(100, "✅ Анализ завершен!")
            time.sleep(1)
            progress_bar.empty()
            
            st.success("🎉 Анализ успешно завершен! Перейдите на вкладку 'Результаты'.")
            if st.button("📊 Посмотреть результаты", type="primary"):
                st.session_state.current_page = 'Результаты'
                st.rerun()

        except Exception as e:
            progress_bar.empty()
            st.error(f"Произошла ошибка во время анализа: {str(e)}")
            st.exception(e)

    def _render_results_page(self):
        """Отрисовка страницы с результатами анализа."""
        if not st.session_state.analysis_completed:
            st.warning("⚠️ Сначала выполните анализ на странице 'Анализ изображения'.")
            return
        
        st.header("📊 Результаты анализа креатива")
        
        col1, col2, col3, col4 = st.columns(4)
        predictions = st.session_state.predictions
        
        with col1:
            st.metric("CTR", f"{predictions.get('ctr', 0) * 100:.2f}%")
        with col2:
            st.metric("Конверсия", f"{predictions.get('conversion_rate', 0) * 100:.2f}%")
        with col3:
            st.metric("Вовлеченность", f"{predictions.get('engagement', 0) * 100:.2f}%")
        with col4:
            explanation = self.ml_engine.explain_prediction(st.session_state.image_features)
            st.metric("Общая оценка", explanation['performance_category'])
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🔮 Предсказания", "🎨 Цветовой анализ", "📐 Композиция", "📝 Текст"])
        
        analysis_results = st.session_state.analysis_results
        
        with tab1:
            st.subheader("Предсказания эффективности и ключевые факторы")
            pred_fig = self.visualizer.plot_performance_prediction(predictions, analysis_results.get('confidence_intervals'))
            st.plotly_chart(pred_fig, use_container_width=True)
            feature_importance = self.ml_engine.get_feature_importance('ctr')
            if feature_importance:
                importance_fig = self.visualizer.plot_feature_importance(feature_importance)
                st.plotly_chart(importance_fig, use_container_width=True)
        
        with tab2:
            st.subheader("Анализ цветовых характеристик")
            if 'color_analysis' in analysis_results:
                color_fig = self.visualizer.plot_color_analysis(analysis_results['color_analysis'])
                st.plotly_chart(color_fig, use_container_width=True)
        
        with tab3:
            st.subheader("Анализ композиции")
            if 'composition_analysis' in analysis_results:
                comp_fig = self.visualizer.plot_composition_analysis(analysis_results['composition_analysis'])
                st.plotly_chart(comp_fig, use_container_width=True)

        with tab4:
            st.subheader("Анализ текстовых элементов")
            if 'text_analysis' in analysis_results:
                text_data = analysis_results['text_analysis']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Текстовых блоков", text_data.get('text_amount', 0))
                    st.metric("Символов", text_data.get('total_characters', 0))
                    cta_status = "✅ Есть" if text_data.get('has_cta', False) else "❌ Нет"
                    st.metric("Призыв к действию", cta_status)
                with col2:
                    st.progress(text_data.get('readability_score', 0), text=f"Читаемость: {text_data.get('readability_score', 0):.2f}")
                    st.progress(text_data.get('text_hierarchy', 0), text=f"Иерархия: {text_data.get('text_hierarchy', 0):.2f}")
                    st.progress(text_data.get('text_contrast', 0), text=f"Контраст текста: {text_data.get('text_contrast', 0):.2f}")

        st.markdown("---")
        if st.button("💡 Посмотреть рекомендации", type="primary"):
            st.session_state.current_page = 'Рекомендации'
            st.rerun()

    def _render_recommendations_page(self):
        """Отрисовка страницы рекомендаций."""
        if not st.session_state.analysis_completed:
            st.warning("⚠️ Сначала выполните анализ на странице 'Анализ изображения'.")
            return
        
        st.header("💡 Рекомендации по улучшению креатива")
        recommendations = st.session_state.recommendations
        if not recommendations:
            st.success("🎉 Отличная работа! Критических рекомендаций не найдено.")
            return

        for rec in recommendations:
            priority_class = f"recommendation-{rec.priority}"
            priority_emoji = {'high': '🔥', 'medium': '⚡', 'low': '💡'}.get(rec.priority, '💡')
            
            with st.container():
                st.markdown(f'<div class="{priority_class}">', unsafe_allow_html=True)
                st.markdown(f"<h5>{priority_emoji} {rec.title} (Влияние: {rec.expected_impact:.1%})</h5>", unsafe_allow_html=True)
                st.write(rec.description)
                with st.expander("Пошаговые действия"):
                    for step in rec.actionable_steps:
                        st.write(f"- {step}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

    def _render_about_page(self):
        """Отрисовка страницы 'О проекте'."""
        st.header("ℹ️ О проекте Creative Performance Predictor")
        st.markdown(f"**Версия {APP_VERSION}**\n\n"
                    "Это интеллектуальная система, которая использует компьютерное зрение и машинное обучение для анализа "
                    "рекламных креативов и предсказания их эффективности.\n\n"
                    "### 🛠 Технологический стек\n"
                    "- **Frontend:** Streamlit\n- **Machine Learning:** Scikit-learn\n"
                    "- **Computer Vision:** OpenCV, Pillow\n- **Data Processing:** Pandas, NumPy\n"
                    "- **Visualization:** Plotly")
        
        if 'training_results' in st.session_state:
            with st.expander("📊 Метрики качества модели (R² score)"):
                st.write(st.session_state.training_results)

    def _create_demo_image(self) -> Optional[Image.Image]:
        """Создание демонстрационного изображения."""
        try:
            width, height = 800, 600
            image = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(image)
            
            for y in range(height):
                r = int(255 * (1 - y/height * 0.2))
                g = int(200 * (1 - y/height * 0.5))
                b = 50
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            font_large = ImageFont.load_default(size=50)
            font_medium = ImageFont.load_default(size=25)

            text_main = "DEMO CREATIVE"
            bbox = draw.textbbox((0, 0), text_main, font=font_large)
            x = (width - (bbox[2] - bbox[0])) / 2
            y = height / 3
            draw.text((x, y), text_main, fill="white", font=font_large)
            
            button_x, button_y = width / 2, height - 100
            button_w, button_h = 200, 60
            draw.rectangle([button_x - button_w/2, button_y - button_h/2, button_x + button_w/2, button_y + button_h/2], fill="red")
            
            cta_text = "BUY NOW"
            cta_bbox = draw.textbbox((0, 0), cta_text, font=font_medium)
            cta_x = button_x - (cta_bbox[2] - cta_bbox[0]) / 2
            cta_y = button_y - (cta_bbox[3] - cta_bbox[1]) / 2
            draw.text((cta_x, cta_y), cta_text, fill="white", font=font_medium)
            
            return image
        except Exception as e:
            st.error(f"Не удалось создать демо-изображение: {e}")
            return None


if __name__ == "__main__":
    if DEPENDENCIES_OK:
        app = CreativePerformanceApp()
        app.run()
