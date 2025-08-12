# image_analyzer.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""
Модуль анализа изображений для Creative Performance Predictor.
Полностью переписанная версия с реальным компьютерным зрением и анализом.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageStat
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
from collections import Counter
import re
from typing import Dict, List, Tuple, Optional, Any
import warnings
import math
import io
warnings.filterwarnings('ignore')

# Продвинутые импорты с fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV недоступен. Используем упрощенные алгоритмы.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("✅ EasyOCR доступен - будет использован для анализа текста")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR недоступен. Попытка использования Tesseract...")
    try:
        import pytesseract
        TESSERACT_AVAILABLE = True
        print("✅ Tesseract доступен как fallback")
    except ImportError:
        TESSERACT_AVAILABLE = False
        print("Warning: OCR недоступен. Будет использована эвристика.")

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
    print("✅ YOLO доступен для детекции объектов")
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO недоступен. Детекция объектов отключена.")

from config import COLOR_ANALYSIS, TEXT_ANALYSIS, get_color_name

class AdvancedImageAnalyzer:
    """
    Продвинутый класс для комплексного анализа изображений креативов.
    Использует современные методы компьютерного зрения и машинного обучения.
    """
    
    def __init__(self):
        self.image: Optional[Image.Image] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.image_hsv: Optional[np.ndarray] = None
        self.image_gray: Optional[np.ndarray] = None
        self.features: Dict[str, Any] = {}
        
        # Инициализация OCR
        self.ocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en', 'ru'], gpu=False)
                print("✅ EasyOCR инициализирован")
            except Exception as e:
                print(f"EasyOCR initialization failed: {e}")
                self.ocr_reader = None
        
        # Инициализация YOLO
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                # Используем nano модель для скорости
                self.yolo_model = YOLO('yolov8n.pt')
                print("✅ YOLO модель загружена")
            except Exception as e:
                print(f"YOLO initialization failed: {e}")
                self.yolo_model = None
        
        # Кэш для анализа
        self.analysis_cache = {}
        
    def load_image(self, image_data) -> bool:
        """Загрузить изображение для анализа с расширенной предобработкой."""
        try:
            if isinstance(image_data, Image.Image):
                self.image = image_data.convert('RGB')
            else:
                self.image = Image.fromarray(image_data).convert('RGB')
            
            # Преобразования для анализа
            self.image_rgb = np.array(self.image)
            
            if CV2_AVAILABLE:
                self.image_hsv = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2HSV)
                self.image_gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
            else:
                self.image_hsv = self._rgb_to_hsv_numpy(self.image_rgb)
                self.image_gray = np.dot(self.image_rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            
            # Очистка кэша при загрузке нового изображения
            self.analysis_cache.clear()
            
            return True
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return False
    
    def analyze_colors(self) -> Dict:
        """Продвинутый анализ цветовых характеристик."""
        if 'color_analysis' in self.analysis_cache:
            return self.analysis_cache['color_analysis']
        
        if self.image_rgb is None or self.image_hsv is None:
            return {}
        
        # Получение доминирующих цветов с улучшенным алгоритмом
        dominant_colors = self._get_dominant_colors_advanced()
        
        # Расширенный анализ цветовых характеристик
        result = {
            'dominant_colors': dominant_colors,
            'harmony_score': self._calculate_color_harmony_advanced(dominant_colors),
            'contrast_score': self._calculate_contrast_advanced(),
            'color_temperature': self._calculate_color_temperature_advanced(),
            'saturation': self._calculate_average_saturation(),
            'brightness': self._calculate_average_brightness(),
            'color_diversity': self._calculate_color_diversity(dominant_colors),
            'warm_cool_ratio': self._calculate_warm_cool_ratio_advanced(dominant_colors),
            'color_balance': self._calculate_color_balance(),
            'color_vibrancy': self._calculate_color_vibrancy(),
            'emotional_impact': self._assess_emotional_color_impact(dominant_colors)
        }
        
        self.analysis_cache['color_analysis'] = result
        return result
    
    def analyze_composition(self) -> Dict:
        """Продвинутый анализ композиционных характеристик."""
        if 'composition_analysis' in self.analysis_cache:
            return self.analysis_cache['composition_analysis']
        
        if self.image_gray is None:
            return {}
        
        # Детекция объектов для анализа композиции
        objects_data = self._detect_objects()
        
        result = {
            'rule_of_thirds_score': self._analyze_rule_of_thirds_advanced(objects_data),
            'visual_balance_score': self._calculate_visual_balance_advanced(objects_data),
            'composition_complexity': self._calculate_composition_complexity_advanced(),
            'center_focus_score': self._analyze_center_focus_advanced(objects_data),
            'leading_lines_score': self._detect_leading_lines_advanced(),
            'symmetry_score': self._calculate_symmetry_advanced(),
            'depth_perception': self._analyze_depth_cues_advanced(),
            'golden_ratio_score': self._analyze_golden_ratio(),
            'visual_flow': self._analyze_visual_flow(objects_data),
            'negative_space': self._analyze_negative_space(),
            'focal_points': len(objects_data.get('high_confidence_objects', [])),
            'composition_dynamics': self._analyze_composition_dynamics(objects_data)
        }
        
        self.analysis_cache['composition_analysis'] = result
        return result
    
    def analyze_text(self) -> Dict:
        """Анализ текстовых элементов."""
        if 'text_analysis' in self.analysis_cache:
            return self.analysis_cache['text_analysis']
        
        if self.image_rgb is None:
            return {}
        
        # Используем лучший доступный OCR
        text_data = self._extract_text_advanced()
        
        if not text_data['texts']:
            # Если текст не найден, возвращаем нули, а не константы
            result = {
                'text_amount': 0,
                'total_characters': 0,
                'readability_score': 1.0,  # Хорошо, если текста нет
                'text_hierarchy': 1.0,
                'text_positioning': 1.0,
                'text_contrast': 1.0,
                'has_cta': False,
                'text_to_image_ratio': 0.0,
                'font_variety': 0,
                'text_density': 0.0,
                'text_coverage': 0.0
            }
        else:
            result = {
                'text_amount': len(text_data['texts']),
                'total_characters': sum(len(t) for t in text_data['texts']),
                'readability_score': self._calculate_text_readability_advanced(text_data),
                'text_hierarchy': self._analyze_text_hierarchy_advanced(text_data),
                'text_positioning': self._analyze_text_positioning_advanced(text_data),
                'text_contrast': self._calculate_text_contrast_advanced(text_data),
                'has_cta': self._detect_cta_advanced(text_data['texts']),
                'text_to_image_ratio': self._calculate_text_to_image_ratio_advanced(text_data),
                'font_variety': self._estimate_font_variety(text_data),
                'text_density': self._calculate_text_density(text_data),
                'text_coverage': self._calculate_text_coverage(text_data)
            }
        
        self.analysis_cache['text_analysis'] = result
        return result
    
    def get_all_features(self) -> Dict[str, Any]:
        """Получение всех извлеченных признаков для ML."""
        features = {}
        
        # Цветовые характеристики
        color_analysis = self.analyze_colors()
        features.update({
            'brightness': color_analysis.get('brightness', 0.5),
            'saturation': color_analysis.get('saturation', 0.5),
            'contrast_score': color_analysis.get('contrast_score', 0.5),
            'color_temperature': color_analysis.get('color_temperature', 0.5),
            'harmony_score': color_analysis.get('harmony_score', 0.5),
            'color_vibrancy': color_analysis.get('color_vibrancy', 0.5),
            'emotional_impact': color_analysis.get('emotional_impact', 0.5)
        })
        
        # Композиционные характеристики
        composition_analysis = self.analyze_composition()
        features.update({
            'rule_of_thirds_score': composition_analysis.get('rule_of_thirds_score', 0.5),
            'visual_balance_score': composition_analysis.get('visual_balance_score', 0.5),
            'composition_complexity': composition_analysis.get('composition_complexity', 0.5),
            'center_focus_score': composition_analysis.get('center_focus_score', 0.5),
            'symmetry_score': composition_analysis.get('symmetry_score', 0.5),
            'negative_space': composition_analysis.get('negative_space', 0.5),
            'visual_flow': composition_analysis.get('visual_flow', 0.5),
            'focal_points': composition_analysis.get('focal_points', 0),
            'overall_complexity': (composition_analysis.get('composition_complexity', 0.5) + 
                                 color_analysis.get('color_diversity', 0.5)) / 2
        })
        
        # Текстовые характеристики
        text_analysis = self.analyze_text()
        features.update({
            'text_amount': min(text_analysis.get('text_amount', 0) / 6.0, 1.0),  # Нормализация
            'readability_score': text_analysis.get('readability_score', 1.0),
            'text_hierarchy': text_analysis.get('text_hierarchy', 1.0),
            'text_contrast': text_analysis.get('text_contrast', 1.0),
            'has_cta': float(text_analysis.get('has_cta', False)),
            'text_positioning': text_analysis.get('text_positioning', 1.0)
        })
        
        # Дополнительные характеристики
        if self.image:
            width, height = self.image.size
            features['aspect_ratio'] = min(width / height, 3.0) / 3.0  # Нормализация
        else:
            features['aspect_ratio'] = 0.5
        
        return features
    
    # === ПРОДВИНУТЫЕ МЕТОДЫ АНАЛИЗА ЦВЕТОВ ===
    
    def _get_dominant_colors_advanced(self, n_colors: int = 8) -> List[Tuple[int, int, int]]:
        """Получение доминирующих цветов с улучшенным алгоритмом."""
        if self.image_rgb is None:
            return []
        
        try:
            # Уменьшаем изображение для ускорения
            height, width = self.image_rgb.shape[:2]
            if width > 300:
                scale = 300 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                if CV2_AVAILABLE:
                    resized = cv2.resize(self.image_rgb, (new_width, new_height))
                else:
                    resized_img = self.image.resize((new_width, new_height))
                    resized = np.array(resized_img)
            else:
                resized = self.image_rgb
            
            # Преобразуем в список пикселей
            pixels = resized.reshape(-1, 3)
            
            # Удаляем очень темные и очень светлые пиксели (шум)
            brightness = np.mean(pixels, axis=1)
            mask = (brightness > 20) & (brightness < 235)
            filtered_pixels = pixels[mask]
            
            if len(filtered_pixels) < 10:
                filtered_pixels = pixels
            
            # K-means кластеризация
            kmeans = KMeans(n_clusters=min(n_colors, len(filtered_pixels)), 
                          random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            
            # Сортируем по распространенности
            labels = kmeans.labels_
            color_counts = Counter(labels)
            sorted_colors = [colors[label] for label, _ in color_counts.most_common()]
            
            return [tuple(color) for color in sorted_colors]
            
        except Exception as e:
            print(f"Ошибка в анализе доминирующих цветов: {e}")
            # Fallback: простое усреднение по регионам
            return self._get_dominant_colors_fallback()
    
    def _get_dominant_colors_fallback(self) -> List[Tuple[int, int, int]]:
        """Простой fallback для получения доминирующих цветов."""
        if self.image_rgb is None:
            return []
        
        # Разбиваем изображение на сетку и усредняем цвета
        height, width = self.image_rgb.shape[:2]
        grid_size = 4
        colors = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                y1 = height * i // grid_size
                y2 = height * (i + 1) // grid_size
                x1 = width * j // grid_size
                x2 = width * (j + 1) // grid_size
                
                region = self.image_rgb[y1:y2, x1:x2]
                avg_color = np.mean(region, axis=(0, 1)).astype(int)
                colors.append(tuple(avg_color))
        
        # Удаляем дубликаты
        unique_colors = []
        for color in colors:
            is_unique = True
            for existing in unique_colors:
                if np.linalg.norm(np.array(color) - np.array(existing)) < 30:
                    is_unique = False
                    break
            if is_unique:
                unique_colors.append(color)
        
        return unique_colors[:6]
    
    def _calculate_color_harmony_advanced(self, colors: List[Tuple[int, int, int]]) -> float:
        """Расчет цветовой гармонии на основе теории цвета."""
        if not colors or len(colors) < 2:
            return 0.5
        
        try:
            # Конвертируем в HSV для анализа
            hsv_colors = []
            for r, g, b in colors:
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                hsv_colors.append((h * 360, s, v))
            
            harmony_score = 0.0
            comparisons = 0
            
            for i, (h1, s1, v1) in enumerate(hsv_colors):
                for j, (h2, s2, v2) in enumerate(hsv_colors[i+1:], i+1):
                    # Анализируем цветовые отношения
                    hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
                    
                    # Комплементарные цвета (противоположные)
                    if 160 <= hue_diff <= 200:
                        harmony_score += 0.9
                    # Триадические (120 градусов)
                    elif 100 <= hue_diff <= 140:
                        harmony_score += 0.8
                    # Аналогичные (соседние)
                    elif hue_diff <= 30:
                        harmony_score += 0.7
                    # Тетрадические (90 градусов)
                    elif 70 <= hue_diff <= 110:
                        harmony_score += 0.6
                    # Другие отношения
                    else:
                        harmony_score += 0.3
                    
                    comparisons += 1
            
            return harmony_score / comparisons if comparisons > 0 else 0.5
            
        except Exception as e:
            print(f"Ошибка в расчете гармонии: {e}")
            return 0.5
    
    def _calculate_contrast_advanced(self) -> float:
        """Расчет контрастности изображения."""
        if self.image_gray is None:
            return 0.5
        
        try:
            # Метод 1: Стандартное отклонение яркости
            std_dev = np.std(self.image_gray) / 255.0
            
            # Метод 2: RMS контраст
            mean_val = np.mean(self.image_gray)
            rms_contrast = np.sqrt(np.mean((self.image_gray - mean_val) ** 2)) / 255.0
            
            # Метод 3: Michelson контраст на краях
            if CV2_AVAILABLE:
                edges = cv2.Canny(self.image_gray, 50, 150)
                edge_contrast = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255.0)
            else:
                edge_contrast = 0.0
            
            # Комбинируем метрики
            contrast_score = (std_dev * 0.4 + rms_contrast * 0.4 + edge_contrast * 0.2)
            return min(contrast_score, 1.0)
            
        except Exception as e:
            print(f"Ошибка в расчете контраста: {e}")
            return 0.5
    
    def _calculate_color_temperature_advanced(self) -> float:
        """Расчет цветовой температуры (теплые vs холодные тона)."""
        if self.image_rgb is None:
            return 0.5
        
        try:
            # Анализируем соотношение теплых и холодных тонов
            warm_pixels = 0
            cool_pixels = 0
            total_pixels = 0
            
            # Сэмплируем каждый 10-й пиксель для скорости
            height, width = self.image_rgb.shape[:2]
            for y in range(0, height, 10):
                for x in range(0, width, 10):
                    r, g, b = self.image_rgb[y, x]
                    
                    # Конвертируем в HSV
                    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                    hue = h * 360
                    
                    # Классификация теплых/холодных тонов
                    if (hue >= 0 and hue <= 60) or (hue >= 300 and hue <= 360):  # Красно-желтый
                        warm_pixels += 1
                    elif hue >= 180 and hue <= 300:  # Сине-зеленый
                        cool_pixels += 1
                    # Остальные считаем нейтральными
                    
                    total_pixels += 1
            
            if total_pixels == 0:
                return 0.5
            
            # Возвращаем соотношение (0 = холодные, 1 = теплые)
            warm_ratio = warm_pixels / total_pixels
            return warm_ratio
            
        except Exception as e:
            print(f"Ошибка в расчете цветовой температуры: {e}")
            return 0.5
    
    def _calculate_average_saturation(self) -> float:
        """Расчет средней насыщенности."""
        if self.image_hsv is None:
            return 0.5
        
        try:
            if CV2_AVAILABLE:
                # OpenCV HSV: S канал в диапазоне 0-255
                saturation = self.image_hsv[:, :, 1] / 255.0
            else:
                # Наш HSV: S канал в диапазоне 0-1
                saturation = self.image_hsv[:, :, 1]
            
            return float(np.mean(saturation))
            
        except Exception as e:
            print(f"Ошибка в расчете насыщенности: {e}")
            return 0.5
    
    def _calculate_average_brightness(self) -> float:
        """Расчет средней яркости."""
        if self.image_hsv is None:
            return 0.5
        
        try:
            if CV2_AVAILABLE:
                # OpenCV HSV: V канал в диапазоне 0-255
                brightness = self.image_hsv[:, :, 2] / 255.0
            else:
                # Наш HSV: V канал в диапазоне 0-1
                brightness = self.image_hsv[:, :, 2]
            
            return float(np.mean(brightness))
            
        except Exception as e:
            print(f"Ошибка в расчете яркости: {e}")
            return 0.5
    
    def _calculate_color_diversity(self, colors: List[Tuple[int, int, int]]) -> float:
        """Расчет цветового разнообразия."""
        if not colors:
            return 0.0
        
        try:
            # Нормализация количества цветов
            diversity = len(colors) / 10.0  # Максимум 10 цветов
            
            # Дополнительно учитываем различия между цветами
            if len(colors) > 1:
                total_distance = 0
                comparisons = 0
                
                for i, color1 in enumerate(colors):
                    for color2 in colors[i+1:]:
                        # Евклидово расстояние в RGB пространстве
                        distance = np.linalg.norm(np.array(color1) - np.array(color2))
                        total_distance += distance
                        comparisons += 1
                
                if comparisons > 0:
                    avg_distance = total_distance / comparisons
                    # Нормализуем (максимальное расстояние в RGB = ~441)
                    diversity += (avg_distance / 441.0) * 0.5
            
            return min(diversity, 1.0)
            
        except Exception as e:
            print(f"Ошибка в расчете цветового разнообразия: {e}")
            return 0.5
    
    def _calculate_warm_cool_ratio_advanced(self, colors: List[Tuple[int, int, int]]) -> float:
        """Расчет соотношения теплых и холодных цветов."""
        if not colors:
            return 0.5
        
        try:
            warm_count = 0
            cool_count = 0
            
            for r, g, b in colors:
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                hue = h * 360
                
                if (hue >= 0 and hue <= 60) or (hue >= 300 and hue <= 360):
                    warm_count += 1
                elif hue >= 180 and hue <= 300:
                    cool_count += 1
            
            total = warm_count + cool_count
            if total == 0:
                return 0.5
            
            return warm_count / total
            
        except Exception as e:
            print(f"Ошибка в расчете warm/cool ratio: {e}")
            return 0.5
    
    def _calculate_color_balance(self) -> float:
        """Расчет цветового баланса."""
        if self.image_rgb is None:
            return 0.5
        
        try:
            # Анализируем баланс RGB каналов
            mean_r = np.mean(self.image_rgb[:, :, 0])
            mean_g = np.mean(self.image_rgb[:, :, 1])
            mean_b = np.mean(self.image_rgb[:, :, 2])
            
            # Рассчитываем отклонения от серого (идеального баланса)
            avg_all = (mean_r + mean_g + mean_b) / 3
            
            deviation_r = abs(mean_r - avg_all) / 255.0
            deviation_g = abs(mean_g - avg_all) / 255.0
            deviation_b = abs(mean_b - avg_all) / 255.0
            
            # Общее отклонение (меньше = лучше баланс)
            total_deviation = (deviation_r + deviation_g + deviation_b) / 3
            
            # Инвертируем для получения оценки баланса
            balance_score = 1.0 - total_deviation
            
            return max(balance_score, 0.0)
            
        except Exception as e:
            print(f"Ошибка в расчете цветового баланса: {e}")
            return 0.5
    
    def _calculate_color_vibrancy(self) -> float:
        """Расчет живости цветов."""
        if self.image_hsv is None:
            return 0.5
        
        try:
            if CV2_AVAILABLE:
                saturation = self.image_hsv[:, :, 1] / 255.0
                brightness = self.image_hsv[:, :, 2] / 255.0
            else:
                saturation = self.image_hsv[:, :, 1]
                brightness = self.image_hsv[:, :, 2]
            
            # Живость = комбинация насыщенности и яркости
            vibrancy = saturation * brightness
            
            return float(np.mean(vibrancy))
            
        except Exception as e:
            print(f"Ошибка в расчете живости цветов: {e}")
            return 0.5
    
    def _assess_emotional_color_impact(self, colors: List[Tuple[int, int, int]]) -> float:
        """Оценка эмоционального воздействия цветовой палитры."""
        if not colors:
            return 0.5
        
        try:
            emotional_scores = []
            
            for r, g, b in colors:
                # Анализируем эмоциональное воздействие каждого цвета
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                hue = h * 360
                
                # Эмоциональные оценки по тонам
                if 0 <= hue <= 30 or 330 <= hue <= 360:  # Красный
                    emotional_scores.append(0.9 * s)  # Высокая эмоциональность
                elif 30 <= hue <= 60:  # Оранжевый
                    emotional_scores.append(0.8 * s)
                elif 60 <= hue <= 90:  # Желтый
                    emotional_scores.append(0.7 * s)
                elif 240 <= hue <= 280:  # Фиолетовый
                    emotional_scores.append(0.6 * s)
                elif 300 <= hue <= 330:  # Пурпурный
                    emotional_scores.append(0.7 * s)
                else:  # Зеленый, синий
                    emotional_scores.append(0.4 * s)
            
            return float(np.mean(emotional_scores)) if emotional_scores else 0.5
            
        except Exception as e:
            print(f"Ошибка в оценке эмоционального воздействия: {e}")
            return 0.5
    
    # === ПРОДВИНУТЫЕ МЕТОДЫ АНАЛИЗА КОМПОЗИЦИИ ===
    
    def _detect_objects(self) -> Dict[str, Any]:
        """Детекция объектов с помощью YOLO."""
        objects_data = {
            'high_confidence_objects': [],
            'all_detections': [],
            'face_detected': False,
            'people_count': 0,
            'main_subjects': []
        }
        
        if not YOLO_AVAILABLE or self.yolo_model is None or self.image is None:
            return objects_data
        
        try:
            # Конвертируем PIL в формат для YOLO
            image_array = np.array(self.image)
            
            # Запускаем детекцию
            results = self.yolo_model(image_array, conf=0.3, verbose=False)
            
            if results and len(results) > 0:
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        
                        for i, box in enumerate(boxes):
                            if hasattr(box, 'conf') and hasattr(box, 'cls'):
                                confidence = float(box.conf[0])
                                class_id = int(box.cls[0])
                                
                                if confidence > 0.5:  # Высокая уверенность
                                    # Получаем координаты bbox
                                    if hasattr(box, 'xyxy'):
                                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                                        
                                        detection = {
                                            'class_id': class_id,
                                            'confidence': confidence,
                                            'bbox': [x1, y1, x2, y2],
                                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                                            'area': (x2 - x1) * (y2 - y1)
                                        }
                                        
                                        objects_data['all_detections'].append(detection)
                                        
                                        if confidence > 0.7:
                                            objects_data['high_confidence_objects'].append(detection)
                                        
                                        # Проверяем на людей (class_id = 0 в COCO)
                                        if class_id == 0:
                                            objects_data['people_count'] += 1
                                            objects_data['face_detected'] = True
                                        
                                        # Основные объекты для композиции
                                        if confidence > 0.6:
                                            objects_data['main_subjects'].append(detection)
            
        except Exception as e:
            print(f"Ошибка в детекции объектов: {e}")
        
        return objects_data
    
    def _analyze_rule_of_thirds_advanced(self, objects_data: Dict) -> float:
        """Анализ соблюдения правила третей с учетом объектов."""
        if self.image is None:
            return 0.5
        
        try:
            width, height = self.image.size
            
            # Линии третей
            third_x1, third_x2 = width / 3, 2 * width / 3
            third_y1, third_y2 = height / 3, 2 * height / 3
            
            # Точки пересечения (точки силы)
            power_points = [
                (third_x1, third_y1), (third_x2, third_y1),
                (third_x1, third_y2), (third_x2, third_y2)
            ]
            
            score = 0.0
            
            # Анализируем расположение объектов
            detected_objects = objects_data.get('high_confidence_objects', [])
            
            if detected_objects:
                for obj in detected_objects:
                    center_x, center_y = obj['center']
                    
                    # Проверяем близость к точкам силы
                    for px, py in power_points:
                        distance = np.sqrt((center_x - px)**2 + (center_y - py)**2)
                        # Нормализуем расстояние
                        max_distance = np.sqrt(width**2 + height**2) / 4
                        proximity = 1.0 - min(distance / max_distance, 1.0)
                        score += proximity * 0.25  # Каждая точка силы дает до 0.25
                
                # Нормализуем по количеству объектов
                score = score / len(detected_objects)
            else:
                # Если объекты не найдены, анализируем визуальные центры
                score = self._analyze_visual_centers_thirds()
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Ошибка в анализе правила третей: {e}")
            return 0.5
    
    def _analyze_visual_centers_thirds(self) -> float:
        """Анализ визуальных центров для правила третей."""
        if self.image_gray is None:
            return 0.5
        
        try:
            height, width = self.image_gray.shape
            
            # Находим области высокого контраста
            if CV2_AVAILABLE:
                edges = cv2.Canny(self.image_gray, 50, 150)
                # Находим контуры
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                centers = []
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Фильтруем маленькие
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            centers.append((cx, cy))
            else:
                # Простой fallback - находим яркие области
                centers = self._find_bright_regions_fallback()
            
            if not centers:
                return 0.5
            
            # Проверяем близость к точкам третей
            third_x1, third_x2 = width / 3, 2 * width / 3
            third_y1, third_y2 = height / 3, 2 * height / 3
            
            power_points = [
                (third_x1, third_y1), (third_x2, third_y1),
                (third_x1, third_y2), (third_x2, third_y2)
            ]
            
            total_score = 0.0
            for cx, cy in centers:
                for px, py in power_points:
                    distance = np.sqrt((cx - px)**2 + (cy - py)**2)
                    max_distance = np.sqrt(width**2 + height**2) / 4
                    proximity = 1.0 - min(distance / max_distance, 1.0)
                    total_score += proximity
            
            return min(total_score / (len(centers) * 4), 1.0)
            
        except Exception as e:
            print(f"Ошибка в анализе визуальных центров: {e}")
            return 0.5
    
    def _find_bright_regions_fallback(self) -> List[Tuple[int, int]]:
        """Поиск ярких областей как fallback."""
        if self.image_gray is None:
            return []
        
        height, width = self.image_gray.shape
        centers = []
        
        # Разбиваем на сетку и находим локальные максимумы
        grid_size = 8
        for i in range(grid_size):
            for j in range(grid_size):
                y1 = height * i // grid_size
                y2 = height * (i + 1) // grid_size
                x1 = width * j // grid_size
                x2 = width * (j + 1) // grid_size
                
                region = self.image_gray[y1:y2, x1:x2]
                if region.size > 0:
                    max_val = np.max(region)
                    if max_val > 150:  # Достаточно яркая область
                        # Находим позицию максимума
                        max_pos = np.unravel_index(np.argmax(region), region.shape)
                        cx = x1 + max_pos[1]
                        cy = y1 + max_pos[0]
                        centers.append((cx, cy))
        
        return centers
    
    def _calculate_visual_balance_advanced(self, objects_data: Dict) -> float:
        """Расчет визуального баланса."""
        if self.image is None:
            return 0.5
        
        try:
            width, height = self.image.size
            center_x, center_y = width / 2, height / 2
            
            # Анализируем баланс на основе объектов
            objects = objects_data.get('high_confidence_objects', [])
            
            if objects:
                left_weight = 0
                right_weight = 0
                top_weight = 0
                bottom_weight = 0
                
                for obj in objects:
                    cx, cy = obj['center']
                    area = obj['area']
                    weight = area * obj['confidence']
                    
                    if cx < center_x:
                        left_weight += weight * (center_x - cx) / center_x
                    else:
                        right_weight += weight * (cx - center_x) / center_x
                    
                    if cy < center_y:
                        top_weight += weight * (center_y - cy) / center_y
                    else:
                        bottom_weight += weight * (cy - center_y) / center_y
                
                # Рассчитываем баланс
                horizontal_balance = 1.0 - abs(left_weight - right_weight) / (left_weight + right_weight + 1e-6)
                vertical_balance = 1.0 - abs(top_weight - bottom_weight) / (top_weight + bottom_weight + 1e-6)
                
                return (horizontal_balance + vertical_balance) / 2
            else:
                # Анализируем баланс яркости
                return self._calculate_brightness_balance()
            
        except Exception as e:
            print(f"Ошибка в расчете визуального баланса: {e}")
            return 0.5
    
    def _calculate_brightness_balance(self) -> float:
        """Расчет баланса яркости."""
        if self.image_gray is None:
            return 0.5
        
        try:
            height, width = self.image_gray.shape
            
            # Разделяем на половины
            left_half = self.image_gray[:, :width//2]
            right_half = self.image_gray[:, width//2:]
            top_half = self.image_gray[:height//2, :]
            bottom_half = self.image_gray[height//2:, :]
            
            # Средняя яркость каждой половины
            left_brightness = np.mean(left_half)
            right_brightness = np.mean(right_half)
            top_brightness = np.mean(top_half)
            bottom_brightness = np.mean(bottom_half)
            
            # Рассчитываем баланс
            horizontal_balance = 1.0 - abs(left_brightness - right_brightness) / 255.0
            vertical_balance = 1.0 - abs(top_brightness - bottom_brightness) / 255.0
            
            return (horizontal_balance + vertical_balance) / 2
            
        except Exception as e:
            print(f"Ошибка в расчете баланса яркости: {e}")
            return 0.5
    
    def _calculate_composition_complexity_advanced(self) -> float:
        """Расчет сложности композиции."""
        if self.image_gray is None:
            return 0.5
        
        try:
            complexity_score = 0.0
            
            # 1. Анализ краев (больше краев = сложнее)
            if CV2_AVAILABLE:
                edges = cv2.Canny(self.image_gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                complexity_score += edge_density * 0.4
            else:
                # Простой анализ градиентов
                grad_x = np.abs(np.diff(self.image_gray.astype(float), axis=1))
                grad_y = np.abs(np.diff(self.image_gray.astype(float), axis=0))
                gradient_density = (np.mean(grad_x) + np.mean(grad_y)) / 255.0
                complexity_score += gradient_density * 0.4
            
            # 2. Анализ текстур
            texture_score = self._calculate_texture_complexity()
            complexity_score += texture_score * 0.3
            
            # 3. Цветовое разнообразие
            color_analysis = self.analyze_colors()
            color_diversity = color_analysis.get('color_diversity', 0.5)
            complexity_score += color_diversity * 0.3
            
            return min(complexity_score, 1.0)
            
        except Exception as e:
            print(f"Ошибка в расчете сложности композиции: {e}")
            return 0.5
    
    def _calculate_texture_complexity(self) -> float:
        """Расчет сложности текстур."""
        if self.image_gray is None:
            return 0.5
        
        try:
            # Локальное стандартное отклонение как мера текстуры
            kernel_size = 9
            height, width = self.image_gray.shape
            
            texture_scores = []
            
            for y in range(0, height - kernel_size, kernel_size):
                for x in range(0, width - kernel_size, kernel_size):
                    window = self.image_gray[y:y+kernel_size, x:x+kernel_size]
                    if window.size > 0:
                        std_dev = np.std(window) / 255.0
                        texture_scores.append(std_dev)
            
            return np.mean(texture_scores) if texture_scores else 0.5
            
        except Exception as e:
            print(f"Ошибка в расчете текстуры: {e}")
            return 0.5
    
    # === ЗАГЛУШКИ ДЛЯ ОСТАЛЬНЫХ МЕТОДОВ ===
    
    def _analyze_center_focus_advanced(self, objects_data: Dict) -> float:
        """Анализ центрального фокуса."""
        if not objects_data.get('high_confidence_objects'):
            return 0.5
        
        try:
            width, height = self.image.size
            center_x, center_y = width / 2, height / 2
            
            total_score = 0.0
            for obj in objects_data['high_confidence_objects']:
                cx, cy = obj['center']
                distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                max_distance = np.sqrt(width**2 + height**2) / 2
                proximity = 1.0 - min(distance / max_distance, 1.0)
                total_score += proximity * obj['confidence']
            
            return min(total_score / len(objects_data['high_confidence_objects']), 1.0)
            
        except:
            return 0.5
    
    def _detect_leading_lines_advanced(self) -> float:
        """Детекция ведущих линий."""
        if not CV2_AVAILABLE or self.image_gray is None:
            return 0.5
        
        try:
            edges = cv2.Canny(self.image_gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                # Нормализуем по размеру изображения
                line_score = min(len(lines) / 10.0, 1.0)
                return line_score
            
            return 0.3
            
        except:
            return 0.5
    
    def _calculate_symmetry_advanced(self) -> float:
        """Расчет симметрии."""
        if self.image_gray is None:
            return 0.5
        
        try:
            height, width = self.image_gray.shape
            
            # Вертикальная симметрия
            left_half = self.image_gray[:, :width//2]
            right_half = np.fliplr(self.image_gray[:, width//2:])
            
            if left_half.shape == right_half.shape:
                vertical_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
                vertical_symmetry = 1.0 - (vertical_diff / 255.0)
            else:
                vertical_symmetry = 0.5
            
            return max(vertical_symmetry, 0.0)
            
        except:
            return 0.5
    
    def _analyze_depth_cues_advanced(self) -> float:
        """Анализ глубины изображения."""
        # Заглушка - можно добавить анализ размытия, перспективы и т.д.
        return 0.5
    
    def _analyze_golden_ratio(self) -> float:
        """Анализ золотого сечения."""
        # Заглушка - анализ пропорций 1:1.618
        return 0.5
    
    def _analyze_visual_flow(self, objects_data: Dict) -> float:
        """Анализ визуального потока."""
        # Заглушка - анализ направления движения взгляда
        return 0.5
    
    def _analyze_negative_space(self) -> float:
        """Анализ негативного пространства."""
        if self.image_gray is None:
            return 0.5
        
        try:
            # Простой анализ - области низкой активности
            threshold = np.mean(self.image_gray)
            empty_space = np.sum(self.image_gray < threshold * 0.8)
            total_space = self.image_gray.shape[0] * self.image_gray.shape[1]
            
            negative_space_ratio = empty_space / total_space
            return min(negative_space_ratio, 1.0)
            
        except:
            return 0.5
    
    def _analyze_composition_dynamics(self, objects_data: Dict) -> float:
        """Анализ динамики композиции."""
        # Заглушка - анализ движения и динамичности
        return 0.5
    
    # === МЕТОДЫ АНАЛИЗА ТЕКСТА ===
    
    def _extract_text_advanced(self) -> Dict[str, Any]:
        """Извлечение текста с помощью OCR."""
        text_data = {
            'texts': [],
            'positions': [],
            'confidences': [],
            'sizes': []
        }
        
        if self.image is None:
            return text_data
        
        try:
            # Пробуем EasyOCR
            if self.ocr_reader is not None:
                results = self.ocr_reader.readtext(np.array(self.image))
                
                for (bbox, text, confidence) in results:
                    if confidence > 0.3 and len(text.strip()) > 1:
                        text_data['texts'].append(text.strip())
                        text_data['positions'].append(bbox)
                        text_data['confidences'].append(confidence)
                        
                        # Размер текста (примерный)
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        width = max(x_coords) - min(x_coords)
                        height = max(y_coords) - min(y_coords)
                        text_data['sizes'].append((width, height))
            
            # Fallback на Tesseract
            elif TESSERACT_AVAILABLE:
                try:
                    text = pytesseract.image_to_string(self.image, lang='rus+eng')
                    if text.strip():
                        text_data['texts'].append(text.strip())
                        text_data['positions'].append([(0, 0), (100, 0), (100, 100), (0, 100)])
                        text_data['confidences'].append(0.5)
                        text_data['sizes'].append((100, 20))
                except:
                    pass
            
        except Exception as e:
            print(f"Ошибка в извлечении текста: {e}")
        
        return text_data
    
    def _calculate_text_readability_advanced(self, text_data: Dict) -> float:
        """Расчет читаемости текста."""
        if not text_data['texts']:
            return 1.0
        
        try:
            readability_scores = []
            
            for i, text in enumerate(text_data['texts']):
                score = 1.0
                
                # Проверяем длину
                if len(text) > 100:
                    score -= 0.2
                
                # Проверяем уверенность OCR
                if i < len(text_data['confidences']):
                    confidence = text_data['confidences'][i]
                    score *= confidence
                
                # Проверяем размер
                if i < len(text_data['sizes']):
                    width, height = text_data['sizes'][i]
                    if height < 10:  # Очень маленький текст
                        score -= 0.3
                
                readability_scores.append(score)
            
            return np.mean(readability_scores)
            
        except:
            return 0.8
    
    def _analyze_text_hierarchy_advanced(self, text_data: Dict) -> float:
        """Анализ иерархии текста."""
        if not text_data['texts'] or len(text_data['texts']) < 2:
            return 1.0
        
        try:
            sizes = text_data.get('sizes', [])
            if not sizes:
                return 0.7
            
            # Анализируем разнообразие размеров
            heights = [size[1] for size in sizes]
            height_variance = np.var(heights)
            
            # Хорошая иерархия имеет умеренное разнообразие размеров
            if height_variance > 50:
                return 0.8
            elif height_variance > 20:
                return 0.9
            else:
                return 0.6
            
        except:
            return 0.7
    
    def _analyze_text_positioning_advanced(self, text_data: Dict) -> float:
        """Анализ позиционирования текста."""
        if not text_data['texts']:
            return 1.0
        
        try:
            positions = text_data.get('positions', [])
            if not positions:
                return 0.7
            
            # Анализируем распределение по изображению
            centers = []
            for bbox in positions:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                centers.append((center_x, center_y))
            
            # Проверяем разумность позиций
            score = 0.8
            
            # Бонус за размещение в читаемых зонах
            width, height = self.image.size
            for cx, cy in centers:
                # Верхняя и нижняя треть - хорошие места для текста
                if cy < height / 3 or cy > 2 * height / 3:
                    score += 0.1
            
            return min(score, 1.0)
            
        except:
            return 0.7
    
    def _calculate_text_contrast_advanced(self, text_data: Dict) -> float:
        """Расчет контрастности текста."""
        if not text_data['texts']:
            return 1.0
        
        try:
            positions = text_data.get('positions', [])
            if not positions:
                return 0.7
            
            contrast_scores = []
            
            for bbox in positions:
                try:
                    # Извлекаем регион с текстом
                    x_coords = [int(point[0]) for point in bbox]
                    y_coords = [int(point[1]) for point in bbox]
                    
                    min_x, max_x = max(0, min(x_coords)), min(self.image.size[0], max(x_coords))
                    min_y, max_y = max(0, min(y_coords)), min(self.image.size[1], max(y_coords))
                    
                    if max_x > min_x and max_y > min_y:
                        # Анализируем контраст в этой области
                        region = self.image_gray[min_y:max_y, min_x:max_x]
                        if region.size > 0:
                            contrast = np.std(region) / 255.0
                            contrast_scores.append(min(contrast * 2, 1.0))
                
                except:
                    contrast_scores.append(0.5)
            
            return np.mean(contrast_scores) if contrast_scores else 0.7
            
        except:
            return 0.7
    
    def _detect_cta_advanced(self, texts: List[str]) -> bool:
        """Детекция призывов к действию."""
        if not texts:
            return False
        
        try:
            cta_keywords = TEXT_ANALYSIS.get('cta_keywords', [])
            
            all_text = ' '.join(texts).lower()
            
            for keyword in cta_keywords:
                if keyword.lower() in all_text:
                    return True
            
            return False
            
        except:
            return False
    
    def _calculate_text_to_image_ratio_advanced(self, text_data: Dict) -> float:
        """Расчет соотношения текста к изображению."""
        if not text_data['texts']:
            return 0.0
        
        try:
            total_text_area = 0
            
            for size in text_data.get('sizes', []):
                width, height = size
                total_text_area += width * height
            
            image_area = self.image.size[0] * self.image.size[1]
            ratio = min(total_text_area / image_area, 1.0)
            
            return ratio
            
        except:
            return 0.1
    
    def _estimate_font_variety(self, text_data: Dict) -> int:
        """Оценка разнообразия шрифтов."""
        sizes = text_data.get('sizes', [])
        if not sizes:
            return 0
        
        # Простая оценка на основе разнообразия размеров
        heights = [size[1] for size in sizes]
        unique_heights = len(set([round(h/5)*5 for h in heights]))  # Группируем по 5px
        
        return min(unique_heights, 5)
    
    def _calculate_text_density(self, text_data: Dict) -> float:
        """Расчет плотности текста."""
        if not text_data['texts']:
            return 0.0
        
        total_chars = sum(len(text) for text in text_data['texts'])
        image_area = self.image.size[0] * self.image.size[1]
        
        # Символов на квадратный пиксель (умножаем на 10000 для читаемых значений)
        density = (total_chars * 10000) / image_area
        
        return min(density, 1.0)
    
    def _calculate_text_coverage(self, text_data: Dict) -> float:
        """Расчет покрытия изображения текстом."""
        return self._calculate_text_to_image_ratio_advanced(text_data)
    
    # === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ===
    
    def _rgb_to_hsv_numpy(self, rgb_array: np.ndarray) -> np.ndarray:
        """Конвертация RGB в HSV без OpenCV."""
        rgb_normalized = rgb_array.astype(float) / 255.0
        
        h = np.zeros(rgb_normalized.shape[:2])
        s = np.zeros(rgb_normalized.shape[:2])
        v = np.zeros(rgb_normalized.shape[:2])
        
        for i in range(rgb_normalized.shape[0]):
            for j in range(rgb_normalized.shape[1]):
                r, g, b = rgb_normalized[i, j]
                h_val, s_val, v_val = colorsys.rgb_to_hsv(r, g, b)
                h[i, j] = h_val
                s[i, j] = s_val
                v[i, j] = v_val
        
        return np.stack([h, s, v], axis=2)


# Алиас для обратной совместимости
ImageAnalyzer = AdvancedImageAnalyzer
