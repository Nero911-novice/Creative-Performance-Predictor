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
                'text_hierarchy': self._analyze_text_
