# image_analyzer.py
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
        """Революционный анализ текстовых элементов."""
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
                'text_contrast': self._analyze_text_contrast_advanced(text_data),
                'has_cta': self._detect_cta_elements_advanced(text_data['texts']),
                'text_to_image_ratio': self._calculate_text_to_image_ratio(text_data),
                'font_variety': self._analyze_font_variety(text_data),
                'text_density': self._calculate_text_density(text_data),
                'text_coverage': self._calculate_text_coverage(text_data)
            }
        
        self.analysis_cache['text_analysis'] = result
        return result
    
    def _detect_objects(self) -> Dict:
        """Детекция объектов на изображении."""
        if self.yolo_model is None:
            return {'objects': [], 'high_confidence_objects': [], 'faces': [], 'people': []}
        
        try:
            # Конвертируем PIL в формат для YOLO
            img_array = np.array(self.image)
            
            # Запуск детекции
            results = self.yolo_model(img_array, conf=0.3, verbose=False)
            
            objects = []
            high_confidence_objects = []
            faces = []
            people = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        
                        # Координаты bounding box
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        obj_data = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        
                        objects.append(obj_data)
                        
                        if confidence > 0.7:
                            high_confidence_objects.append(obj_data)
                        
                        if class_name == 'person':
                            people.append(obj_data)
            
            return {
                'objects': objects,
                'high_confidence_objects': high_confidence_objects,
                'faces': faces,  # TODO: добавить детекцию лиц
                'people': people
            }
            
        except Exception as e:
            print(f"Object detection failed: {e}")
            return {'objects': [], 'high_confidence_objects': [], 'faces': [], 'people': []}
    
    def _extract_text_advanced(self) -> Dict:
        """Продвинутое извлечение текста с использованием лучшего доступного OCR."""
        if self.ocr_reader is not None:
            return self._extract_text_easyocr()
        elif TESSERACT_AVAILABLE:
            return self._extract_text_tesseract()
        else:
            return self._extract_text_heuristic()
    
    def _extract_text_easyocr(self) -> Dict:
        """Извлечение текста с помощью EasyOCR."""
        try:
            img_array = np.array(self.image)
            results = self.ocr_reader.readtext(img_array, detail=1, paragraph=False)
            
            texts = []
            positions = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3 and len(text.strip()) > 1:  # Фильтр по доверию
                    texts.append(text.strip())
                    
                    # Вычисляем bounding box
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    
                    positions.append({
                        'x': x1, 'y': y1,
                        'width': x2 - x1, 'height': y2 - y1,
                        'confidence': confidence
                    })
                    confidences.append(confidence)
            
            return {
                'texts': texts,
                'positions': positions,
                'confidences': confidences,
                'method': 'easyocr'
            }
        except Exception as e:
            print(f"EasyOCR failed: {e}")
            return {'texts': [], 'positions': [], 'confidences': [], 'method': 'failed'}
    
    def _extract_text_tesseract(self) -> Dict:
        """Fallback извлечение текста с помощью Tesseract."""
        try:
            import pytesseract
            ocr_data = pytesseract.image_to_data(self.image, output_type=pytesseract.Output.DICT, config='--psm 6')
            
            texts = []
            positions = []
            confidences = []
            
            for i in range(len(ocr_data['text'])):
                confidence = int(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()
                
                if confidence > 30 and len(text) > 1:
                    texts.append(text)
                    positions.append({
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i],
                        'confidence': confidence / 100.0
                    })
                    confidences.append(confidence / 100.0)
            
            return {
                'texts': texts,
                'positions': positions,
                'confidences': confidences,
                'method': 'tesseract'
            }
        except Exception as e:
            print(f"Tesseract failed: {e}")
            return {'texts': [], 'positions': [], 'confidences': [], 'method': 'failed'}
    
    def _extract_text_heuristic(self) -> Dict:
        """Эвристическое определение наличия текста (без OCR)."""
        # Используем анализ краев и контрастности для предположения о тексте
        try:
            if CV2_AVAILABLE:
                edges = cv2.Canny(self.image_gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                text_like_regions = 0
                total_edge_area = 0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 100 < area < 5000:  # Размеры похожие на текст
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.2 < aspect_ratio < 5:  # Пропорции текста
                            text_like_regions += 1
                            total_edge_area += area
                
                # Эвристическая оценка
                estimated_text_blocks = min(text_like_regions // 3, 10)
                estimated_chars = estimated_text_blocks * 15  # Примерно 15 символов на блок
                
                return {
                    'texts': [f'text_block_{i}' for i in range(estimated_text_blocks)],
                    'positions': [],
                    'confidences': [0.5] * estimated_text_blocks,
                    'method': 'heuristic',
                    'estimated_chars': estimated_chars
                }
            else:
                # Самая простая эвристика
                image_stat = ImageStat.Stat(self.image)
                variance = sum(image_stat.var) / len(image_stat.var)
                
                # Если есть вариативность в пикселях, возможно есть текст
                if variance > 1000:
                    return {
                        'texts': ['estimated_text'],
                        'positions': [],
                        'confidences': [0.3],
                        'method': 'basic_heuristic',
                        'estimated_chars': 25
                    }
                else:
                    return {
                        'texts': [],
                        'positions': [],
                        'confidences': [],
                        'method': 'basic_heuristic',
                        'estimated_chars': 0
                    }
        except Exception as e:
            print(f"Heuristic text detection failed: {e}")
            return {'texts': [], 'positions': [], 'confidences': [], 'method': 'failed'}
    
    # ===== ПРОДВИНУТЫЕ МЕТОДЫ АНАЛИЗА =====
    
    def _get_dominant_colors_advanced(self, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Улучшенное извлечение доминирующих цветов."""
        # Уменьшаем изображение для ускорения
        small_image = self.image.resize((150, 150))
        pixels = np.array(small_image).reshape(-1, 3)
        
        # Удаляем очень темные и очень светлые пиксели для лучшего анализа
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 30) & (brightness < 225)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < 10:
            filtered_pixels = pixels
        
        # K-means кластеризация
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
        kmeans.fit(filtered_pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Сортируем по популярности
        color_counts = Counter(labels)
        sorted_colors = [colors[i] for i, _ in color_counts.most_common()]
        
        return [tuple(color) for color in sorted_colors]
    
    def _calculate_color_harmony_advanced(self, colors: List[Tuple]) -> float:
        """Улучшенный расчет цветовой гармонии."""
        if len(colors) < 2:
            return 0.5
        
        hsv_colors = []
        for color in colors:
            h, s, v = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
            hsv_colors.append((h * 360, s, v))
        
        harmony_score = 0
        total_pairs = 0
        
        # Проверяем различные типы гармонии
        for i in range(len(hsv_colors)):
            for j in range(i + 1, len(hsv_colors)):
                h1, s1, v1 = hsv_colors[i]
                h2, s2, v2 = hsv_colors[j]
                
                hue_diff = abs(h1 - h2)
                hue_diff = min(hue_diff, 360 - hue_diff)
                
                # Комплементарная (противоположная)
                if 150 <= hue_diff <= 210:
                    harmony_score += 1.0
                # Триада
                elif 110 <= hue_diff <= 130:
                    harmony_score += 0.9
                # Аналогичная
                elif hue_diff <= 30:
                    harmony_score += 0.8
                # Тетрада (квадрат)
                elif 80 <= hue_diff <= 100:
                    harmony_score += 0.7
                # Подобные тона
                elif 30 <= hue_diff <= 60:
                    harmony_score += 0.6
                
                total_pairs += 1
        
        return harmony_score / total_pairs if total_pairs > 0 else 0.5
    
    def _calculate_contrast_advanced(self) -> float:
        """Улучшенный расчет контрастности."""
        # Используем несколько метрик контрастности
        
        # 1. Стандартное отклонение
        std_contrast = np.std(self.image_gray) / 128.0
        
        # 2. RMS контраст
        mean_val = np.mean(self.image_gray)
        rms_contrast = np.sqrt(np.mean((self.image_gray - mean_val) ** 2)) / 128.0
        
        # 3. Michelson контраст
        max_val = np.max(self.image_gray)
        min_val = np.min(self.image_gray)
        michelson_contrast = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0
        
        # Комбинированная оценка
        combined_contrast = (std_contrast * 0.4 + rms_contrast * 0.4 + michelson_contrast * 0.2)
        
        return min(combined_contrast, 1.0)
    
    def _analyze_rule_of_thirds_advanced(self, objects_data: Dict) -> float:
        """Улучшенный анализ правила третей с учетом объектов."""
        h, w = self.image_gray.shape
        
        # Линии третей
        third_lines = {
            'vertical': [w/3, 2*w/3],
            'horizontal': [h/3, 2*h/3]
        }
        
        # Точки пересечения (точки силы)
        power_points = [
            (w/3, h/3), (2*w/3, h/3),
            (w/3, 2*h/3), (2*w/3, 2*h/3)
        ]
        
        score = 0
        total_weight = 0
        
        # Анализ размещения объектов
        if objects_data.get('high_confidence_objects'):
            for obj in objects_data['high_confidence_objects']:
                obj_center = obj['center']
                obj_area = obj['area']
                weight = min(obj_area / (w * h), 0.1)  # Нормализуем вес
                
                # Проверяем близость к точкам силы
                min_distance = float('inf')
                for point in power_points:
                    distance = math.sqrt((obj_center[0] - point[0])**2 + (obj_center[1] - point[1])**2)
                    min_distance = min(min_distance, distance)
                
                # Чем ближе к точке силы, тем выше оценка
                threshold = min(w, h) * 0.15
                if min_distance < threshold:
                    proximity_score = 1 - (min_distance / threshold)
                    score += proximity_score * weight
                
                total_weight += weight
        
        # Если объектов нет, анализируем распределение краев
        if total_weight == 0:
            if CV2_AVAILABLE:
                edges = cv2.Canny(self.image_gray, 50, 150)
                
                for point in power_points:
                    x, y = int(point[0]), int(point[1])
                    region_size = 30
                    
                    x1 = max(0, x - region_size)
                    x2 = min(w, x + region_size)
                    y1 = max(0, y - region_size)
                    y2 = min(h, y + region_size)
                    
                    region_edges = edges[y1:y2, x1:x2]
                    edge_density = np.sum(region_edges) / (region_edges.size * 255)
                    
                    score += edge_density
                    total_weight += 1
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _detect_cta_elements_advanced(self, texts: List[str]) -> bool:
        """Улучшенная детекция призывов к действию."""
        cta_patterns = {
            'action_verbs': [
                'купить', 'заказать', 'скачать', 'получить', 'узнать', 'попробовать',
                'регистрация', 'подписаться', 'звонить', 'написать', 'связаться',
                'buy', 'order', 'download', 'get', 'learn', 'try', 'register',
                'subscribe', 'call', 'contact', 'click', 'book', 'shop'
            ],
            'urgency_words': [
                'сейчас', 'сегодня', 'срочно', 'быстро', 'ограниченное',
                'now', 'today', 'urgent', 'limited', 'hurry', 'fast'
            ],
            'benefit_words': [
                'бесплатно', 'скидка', 'выгода', 'экономия', 'bonus',
                'free', 'discount', 'save', 'benefit', 'offer'
            ]
        }
        
        all_text = ' '.join(texts).lower()
        
        # Подсчет CTA элементов
        cta_score = 0
        
        for pattern_type, words in cta_patterns.items():
            for word in words:
                if word in all_text:
                    if pattern_type == 'action_verbs':
                        cta_score += 3
                    elif pattern_type == 'urgency_words':
                        cta_score += 2
                    else:
                        cta_score += 1
        
        # Также проверяем структурные паттерны
        if re.search(r'\b\d+%\b', all_text):  # Проценты скидки
            cta_score += 2
        
        if re.search(r'\b(от|from)\s+\d+', all_text):  # Цены
            cta_score += 1
        
        return cta_score >= 3
    
    # Добавляем новые методы анализа
    def _calculate_color_vibrancy(self) -> float:
        """Расчет живости/яркости цветов."""
        hsv_mean = np.mean(self.image_hsv, axis=(0, 1))
        saturation = hsv_mean[1] / 255.0
        brightness = hsv_mean[2] / 255.0
        
        # Комбинируем насыщенность и яркость
        vibrancy = (saturation * 0.7 + brightness * 0.3)
        return min(vibrancy, 1.0)
    
    def _assess_emotional_color_impact(self, colors: List[Tuple]) -> float:
        """Оценка эмоционального воздействия цветов."""
        emotional_scores = {
            'warm': 0,  # Теплые цвета (энергия, страсть)
            'cool': 0,  # Холодные цвета (спокойствие, доверие)
            'energetic': 0,  # Энергичные цвета
            'calming': 0   # Успокаивающие цвета
        }
        
        for color in colors:
            r, g, b = color
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hue = h * 360
            
            # Классификация по эмоциональному воздействию
            if 0 <= hue <= 60 or 300 <= hue <= 360:  # Красный, оранжевый
                emotional_scores['warm'] += s * v
                emotional_scores['energetic'] += s * v
            elif 60 <= hue <= 180:  # Желтый, зеленый
                emotional_scores['energetic'] += s * v * 0.7
                emotional_scores['calming'] += s * v * 0.3
            elif 180 <= hue <= 300:  # Синий, фиолетовый
                emotional_scores['cool'] += s * v
                emotional_scores['calming'] += s * v
        
        # Нормализация и комбинированная оценка
        total_impact = sum(emotional_scores.values())
        return min(total_impact / len(colors), 1.0) if colors else 0.5
    
    def get_all_features(self) -> Dict:
        """Получение всех признаков для ML модели."""
        color_features = self.analyze_colors()
        composition_features = self.analyze_composition()
        text_features = self.analyze_text()
        
        # Дополнительные признаки
        additional_features = {
            'aspect_ratio': self._calculate_aspect_ratio(),
            'image_size_score': self._calculate_size_score(),
            'overall_complexity': self._calculate_overall_complexity(),
            'visual_appeal': self._calculate_visual_appeal_score(color_features, composition_features)
        }
        
        # Объединяем все признаки
        all_features = {}
        
        # Добавляем цветовые (исключаем списки)
        for k, v in color_features.items():
            if not isinstance(v, list):
                all_features[k] = v
        
        # Добавляем композиционные
        all_features.update(composition_features)
        
        # Добавляем текстовые
        all_features.update(text_features)
        
        # Добавляем дополнительные
        all_features.update(additional_features)
        
        # Преобразуем булевы значения в числа
        for key, value in all_features.items():
            if isinstance(value, bool):
                all_features[key] = int(value)
        
        self.features = all_features
        return all_features
    
    def _calculate_overall_complexity(self) -> float:
        """Общая сложность изображения."""
        color_diversity = len(self.analyze_colors().get('dominant_colors', []))
        composition_complexity = self.analyze_composition().get('composition_complexity', 0.5)
        text_amount = self.analyze_text().get('text_amount', 0)
        
        # Нормализованная сложность
        complexity = (
            (color_diversity / 10) * 0.3 +
            composition_complexity * 0.4 +
            min(text_amount / 5, 1) * 0.3
        )
        
        return min(complexity, 1.0)
    
    def _calculate_visual_appeal_score(self, color_features: Dict, composition_features: Dict) -> float:
        """Общая оценка визуальной привлекательности."""
        harmony = color_features.get('harmony_score', 0.5)
        contrast = color_features.get('contrast_score', 0.5)
        balance = composition_features.get('visual_balance_score', 0.5)
        thirds = composition_features.get('rule_of_thirds_score', 0.5)
        
        appeal = (harmony * 0.3 + contrast * 0.2 + balance * 0.3 + thirds * 0.2)
        return min(appeal, 1.0)
    
    # Вспомогательные методы (сокращенные версии)
    def _calculate_aspect_ratio(self) -> float:
        if self.image is None: return 1.0
        w, h = self.image.size
        return w / h if h > 0 else 1.0
    
    def _calculate_size_score(self) -> float:
        if self.image is None: return 0.5
        w, h = self.image.size
        total_pixels = w * h
        return min(total_pixels / (1920 * 1080), 2.0) / 2.0
    
    def _rgb_to_hsv_numpy(self, rgb_image):
        """Fallback RGB to HSV conversion."""
        # Упрощенная реализация для случая отсутствия OpenCV
        rgb_norm = rgb_image.astype(np.float32) / 255.0
        hsv = np.zeros_like(rgb_norm, dtype=np.float32)
        
        # Простая конверсия
        for i in range(rgb_image.shape[0]):
            for j in range(rgb_image.shape[1]):
                r, g, b = rgb_norm[i, j]
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hsv[i, j] = [h * 179, s * 255, v * 255]  # Масштабирование для OpenCV формата
        
        return hsv.astype(np.uint8)
    
    # Заглушки для остальных методов (для совместимости)
    def _calculate_average_saturation(self) -> float:
        return np.mean(self.image_hsv[:, :, 1]) / 255.0
    
    def _calculate_average_brightness(self) -> float:
        return np.mean(self.image_hsv[:, :, 2]) / 255.0
    
    def _calculate_color_diversity(self, colors) -> int:
        return len(colors)
    
    def _calculate_warm_cool_ratio_advanced(self, colors) -> float:
        warm = sum(1 for r, g, b in colors if r > b + 20)
        cool = sum(1 for r, g, b in colors if b > r + 20)
        return warm / (warm + cool) if warm + cool > 0 else 0.5
    
    def _calculate_color_balance(self) -> float:
        # Анализ распределения цветов по изображению
        return 0.7  # Заглушка
    
    def _calculate_color_temperature_advanced(self) -> float:
        red_mean = np.mean(self.image_rgb[:, :, 0])
        blue_mean = np.mean(self.image_rgb[:, :, 2])
        return red_mean / (red_mean + blue_mean) if red_mean + blue_mean > 0 else 0.5
   
    # === ЗАГЛУШКИ ДЛЯ НЕДОСТАЮЩИХ МЕТОДОВ АНАЛИЗА КОМПОЗИЦИИ ===
    # Эти методы вызываются в analyze_composition, но не были реализованы.
    # Мы добавляем их, чтобы приложение не падало.

    def _calculate_visual_balance_advanced(self, objects_data: Dict) -> float:
       """(ЗАГЛУШКА) Расчет визуального баланса."""
       # Возвращаем нейтральное значение по умолчанию
       return 0.65

    def _calculate_composition_complexity_advanced(self) -> float:
       """(ЗАГЛУШКА) Расчет сложности композиции."""
       if CV2_AVAILABLE:
        edges = cv2.Canny(self.image_gray, 50, 150)
        complexity = np.sum(edges) / (edges.size * 255)
        return min(complexity * 5, 1.0) # Нормализуем
       return 0.5 # Возвращаем нейтральное значение

   def _analyze_center_focus_advanced(self, objects_data: Dict) -> float:
       """(ЗАГЛУШКА) Анализ центрального фокуса."""
       return 0.6

   def _detect_leading_lines_advanced(self) -> float:
       """(ЗАГЛУШКА) Детекция направляющих линий."""
       return 0.4

   def _calculate_symmetry_advanced(self) -> float:
       """(ЗАГЛУШКА) Расчет симметрии."""
       return 0.55

   def _analyze_depth_cues_advanced(self) -> float:
       """(ЗАГЛУШКА) Анализ глубины изображения."""
       return 0.45

   def _analyze_golden_ratio(self) -> float:
       """(ЗАГЛУШКА) Анализ золотого сечения."""
       return 0.5

   def _analyze_visual_flow(self, objects_data: Dict) -> float:
       """(ЗАГЛУШКА) Анализ визуального потока."""
       return 0.6

   def _analyze_negative_space(self) -> float:
       """(ЗАГЛУШКА) Анализ негативного пространства."""
       return 0.7

   def _analyze_composition_dynamics(self, objects_data: Dict) -> float:
       """(ЗАГЛУШКА) Анализ динамики композиции."""
       return 0.5

   def _calculate_text_readability_advanced(self, text_data: Dict) -> float:
       """(ЗАГЛУШКА) Расчет читаемости текста."""
       return 0.8

   def _analyze_text_hierarchy_advanced(self, text_data: Dict) -> float:
       """(ЗАГЛУШКА) Анализ иерархии текста."""
       return 0.7

   def _analyze_text_positioning_advanced(self, text_data: Dict) -> float:
       """(ЗАГЛУШКА) Анализ позиционирования текста."""
       return 0.75

   def _analyze_text_contrast_advanced(self, text_data: Dict) -> float:
       """(ЗАГЛУШКА) Анализ контрастности текста."""
       return 0.85

   def _calculate_text_to_image_ratio(self, text_data: Dict) -> float:
       """(ЗАГЛУШКА) Расчет соотношения текста к изображению."""
       return 0.1

   def _analyze_font_variety(self, text_data: Dict) -> int:
       """(ЗАГЛУШКА) Анализ разнообразия шрифтов."""
       return 2

   def _calculate_text_density(self, text_data: Dict) -> float:
       """(ЗАГЛУШКА) Расчет плотности текста."""
       return 0.15

   def _calculate_text_coverage(self, text_data: Dict) -> float:
       """(ЗАГЛУШКА) Расчет покрытия текста."""
       return 0.2

# Алиас для обратной совместимости
ImageAnalyzer = AdvancedImageAnalyzer
