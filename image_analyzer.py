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

   # Методы для замены заглушек в AdvancedImageAnalyzer
# Все методы должны быть добавлены внутрь класса AdvancedImageAnalyzer

def _calculate_symmetry_advanced(self) -> float:
    """
    Расчет симметрии изображения на основе сравнения половин.
    Основано на исследованиях восприятия симметричных композиций.
    """
    try:
        if self.image_gray is None:
            return 0.5
        
        h, w = self.image_gray.shape
        
        # Вертикальная симметрия (левая-правая половины)
        left_half = self.image_gray[:, :w//2]
        right_half = self.image_gray[:, w//2:]
        
        # Отражаем правую половину для сравнения
        right_half_flipped = np.fliplr(right_half)
        
        # Приводим к одинаковому размеру
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_crop = left_half[:, :min_width]
        right_crop = right_half_flipped[:, :min_width]
        
        # Расчет различий
        if CV2_AVAILABLE:
            # Используем структурное сходство
            diff = cv2.absdiff(left_crop, right_crop)
            vertical_symmetry = 1.0 - (np.mean(diff) / 255.0)
        else:
            # Простое сравнение пикселей
            diff = np.abs(left_crop.astype(float) - right_crop.astype(float))
            vertical_symmetry = 1.0 - (np.mean(diff) / 255.0)
        
        # Горизонтальная симметрия (верх-низ)
        top_half = self.image_gray[:h//2, :]
        bottom_half = self.image_gray[h//2:, :]
        
        # Отражаем нижнюю половину
        bottom_half_flipped = np.flipud(bottom_half)
        
        # Приводим к одинаковому размеру
        min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
        top_crop = top_half[:min_height, :]
        bottom_crop = bottom_half_flipped[:min_height, :]
        
        if CV2_AVAILABLE:
            diff = cv2.absdiff(top_crop, bottom_crop)
            horizontal_symmetry = 1.0 - (np.mean(diff) / 255.0)
        else:
            diff = np.abs(top_crop.astype(float) - bottom_crop.astype(float))
            horizontal_symmetry = 1.0 - (np.mean(diff) / 255.0)
        
        # Комбинированная оценка с весами (вертикальная симметрия важнее)
        combined_symmetry = vertical_symmetry * 0.7 + horizontal_symmetry * 0.3
        
        return np.clip(combined_symmetry, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in symmetry calculation: {e}")
        return 0.5

def _analyze_negative_space(self) -> float:
    """
    Анализ негативного пространства - критически важный фактор композиции.
    Основано на принципах дизайна и визуального восприятия.
    """
    try:
        if self.image_gray is None:
            return 0.5
        
        # Бинаризация изображения для выделения объектов
        if CV2_AVAILABLE:
            # Используем адаптивную пороговую обработку
            binary = cv2.adaptiveThreshold(
                self.image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Морфологические операции для очистки
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
        else:
            # Простая пороговая обработка
            mean_intensity = np.mean(self.image_gray)
            binary = (self.image_gray > mean_intensity).astype(np.uint8) * 255
        
        # Расчет соотношения негативного пространства
        total_pixels = binary.size
        negative_pixels = np.sum(binary == 255)  # Белые пиксели = негативное пространство
        negative_ratio = negative_pixels / total_pixels
        
        # Анализ распределения негативного пространства
        if CV2_AVAILABLE:
            # Находим связные компоненты негативного пространства
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Анализ размеров негативных областей
                areas = [cv2.contourArea(contour) for contour in contours]
                largest_area = max(areas) if areas else 0
                
                # Проверяем, не доминирует ли одна большая область
                dominance = largest_area / total_pixels if total_pixels > 0 else 0
                
                # Хорошее негативное пространство: 30-70% изображения, не слишком фрагментированное
                if 0.3 <= negative_ratio <= 0.7:
                    space_quality = 1.0
                elif 0.2 <= negative_ratio <= 0.8:
                    space_quality = 0.8
                else:
                    space_quality = 0.4
                
                # Корректировка на основе доминирования
                if dominance > 0.8:  # Слишком большая единая область
                    space_quality *= 0.7
                elif dominance < 0.1:  # Слишком фрагментированное
                    space_quality *= 0.8
                
            else:
                space_quality = negative_ratio
        else:
            # Упрощенная оценка без контуров
            if 0.3 <= negative_ratio <= 0.7:
                space_quality = 1.0
            elif 0.2 <= negative_ratio <= 0.8:
                space_quality = 0.8
            else:
                space_quality = 0.4
        
        return np.clip(space_quality, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in negative space analysis: {e}")
        return 0.5

def _detect_leading_lines_advanced(self) -> float:
    """
    Детекция направляющих линий с использованием преобразования Хафа.
    Основано на теории визуального потока и направления взгляда.
    """
    try:
        if not CV2_AVAILABLE or self.image_gray is None:
            return 0.4  # Возвращаем средний результат без OpenCV
        
        # Детекция краев
        edges = cv2.Canny(self.image_gray, 50, 150, apertureSize=3)
        
        # Преобразование Хафа для поиска линий
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.2  # Нет выраженных линий
        
        h, w = self.image_gray.shape
        center_x, center_y = w // 2, h // 2
        
        # Анализ найденных линий
        leading_lines_score = 0.0
        diagonal_lines = 0
        convergent_lines = 0
        
        # Определяем точки схождения и углы линий
        angles = []
        
        for line in lines:
            rho, theta = line[0]
            angles.append(theta)
            
            # Вычисляем точки пересечения с границами изображения
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Проверяем, ведет ли линия к центру или важным точкам
            # Расстояние от линии до центра изображения
            dist_to_center = abs(rho - (center_x * a + center_y * b))
            
            # Нормализуем расстояние
            max_dist = max(w, h)
            normalized_dist = dist_to_center / max_dist
            
            # Линии, ведущие к центру, получают высокую оценку
            if normalized_dist < 0.1:  # Близко к центру
                leading_lines_score += 0.3
                convergent_lines += 1
            elif normalized_dist < 0.2:  # Умеренно близко
                leading_lines_score += 0.2
            
            # Диагональные линии более эффективны для направления взгляда
            angle_deg = np.degrees(theta)
            if 30 <= angle_deg <= 60 or 120 <= angle_deg <= 150:
                diagonal_lines += 1
                leading_lines_score += 0.1
        
        # Анализ распределения углов (параллельные и перпендикулярные линии)
        if len(angles) > 1:
            angles_deg = np.degrees(angles)
            
            # Проверяем наличие доминирующих направлений
            angle_hist, _ = np.histogram(angles_deg, bins=18, range=(0, 180))
            dominant_directions = np.sum(angle_hist > 1)  # Направления с несколькими линиями
            
            if dominant_directions >= 2:  # Есть структурированность
                leading_lines_score += 0.2
        
        # Бонус за оптимальное количество линий
        line_count = len(lines)
        if 3 <= line_count <= 8:  # Оптимальное количество
            leading_lines_score += 0.2
        elif line_count > 15:  # Слишком много - создает хаос
            leading_lines_score *= 0.5
        
        # Бонус за конвергентные линии (сходящиеся к точке)
        if convergent_lines >= 2:
            leading_lines_score += 0.3
        
        # Нормализация и ограничение
        final_score = leading_lines_score / max(1, len(lines) * 0.1)
        
        return np.clip(final_score, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in leading lines detection: {e}")
        return 0.4

def _analyze_golden_ratio(self) -> float:
    """
    Анализ соответствия золотому сечению (φ ≈ 1.618).
    Основано на нейроэстетических исследованиях восприятия пропорций.
    """
    try:
        if self.image_gray is None:
            return 0.5
        
        h, w = self.image_gray.shape
        phi = 1.618033988749895  # Золотое сечение
        
        golden_score = 0.0
        
        # 1. Анализ общих пропорций изображения
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Проверяем близость к золотому сечению или его обратному значению
        ratio_diff_phi = abs(aspect_ratio - phi)
        ratio_diff_inv_phi = abs(aspect_ratio - (1/phi))
        
        min_ratio_diff = min(ratio_diff_phi, ratio_diff_inv_phi)
        
        if min_ratio_diff < 0.1:  # Очень близко к золотому сечению
            golden_score += 0.4
        elif min_ratio_diff < 0.2:  # Близко
            golden_score += 0.3
        elif min_ratio_diff < 0.3:  # Умеренно близко
            golden_score += 0.2
        
        # 2. Анализ размещения по золотому сечению
        # Вертикальные линии золотого сечения
        golden_x1 = w / phi  # ~38.2% от ширины
        golden_x2 = w - golden_x1  # ~61.8% от ширины
        
        # Горизонтальные линии золотого сечения
        golden_y1 = h / phi  # ~38.2% от высоты
        golden_y2 = h - golden_y1  # ~61.8% от высоты
        
        # Точки золотого сечения
        golden_points = [
            (golden_x1, golden_y1), (golden_x2, golden_y1),
            (golden_x1, golden_y2), (golden_x2, golden_y2)
        ]
        
        # 3. Анализ размещения ключевых элементов
        if CV2_AVAILABLE:
            # Находим области с высокой контрастностью (потенциальные ключевые элементы)
            edges = cv2.Canny(self.image_gray, 50, 150)
            
            # Разбиваем изображение на сегменты по золотому сечению
            segments_scores = []
            
            for gx, gy in golden_points:
                # Область вокруг точки золотого сечения
                x1, x2 = max(0, int(gx - w*0.05)), min(w, int(gx + w*0.05))
                y1, y2 = max(0, int(gy - h*0.05)), min(h, int(gy + h*0.05))
                
                if x2 > x1 and y2 > y1:
                    region = edges[y1:y2, x1:x2]
                    activity = np.sum(region) / (region.size * 255) if region.size > 0 else 0
                    segments_scores.append(activity)
            
            # Если есть активность в точках золотого сечения
            if segments_scores:
                avg_activity = np.mean(segments_scores)
                max_activity = np.max(segments_scores)
                
                if max_activity > 0.1:  # Есть значимая активность
                    golden_score += 0.3
                if avg_activity > 0.05:  # Общая активность в золотых точках
                    golden_score += 0.2
        
        # 4. Анализ спирали Фибоначчи (упрощенный)
        # Проверяем концентрацию визуального веса по спирали
        spiral_score = self._analyze_fibonacci_spiral()
        golden_score += spiral_score * 0.3
        
        return np.clip(golden_score, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in golden ratio analysis: {e}")
        return 0.5

def _analyze_fibonacci_spiral(self) -> float:
    """Вспомогательный метод для анализа спирали Фибоначчи."""
    try:
        if self.image_gray is None:
            return 0.5
        
        h, w = self.image_gray.shape
        center_x, center_y = w // 2, h // 2
        
        # Создаем точки вдоль упрощенной спирали Фибоначчи
        spiral_points = []
        phi = 1.618033988749895
        
        for i in range(20):  # 20 точек вдоль спирали
            angle = i * 0.5  # Угол в радианах
            radius = (i * 10) / phi  # Радиус, уменьшающийся по золотому сечению
            
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            if 0 <= x < w and 0 <= y < h:
                spiral_points.append((int(x), int(y)))
        
        if not spiral_points:
            return 0.5
        
        # Анализируем интенсивность вдоль спирали
        intensities = []
        for x, y in spiral_points:
            intensities.append(self.image_gray[y, x])
        
        # Проверяем, есть ли градиент или концентрация вдоль спирали
        if len(intensities) > 1:
            intensity_var = np.var(intensities)
            normalized_var = intensity_var / (255 * 255)  # Нормализация
            
            # Умеренная вариация указывает на структурированность
            if 0.1 <= normalized_var <= 0.4:
                return 0.8
            elif 0.05 <= normalized_var <= 0.6:
                return 0.6
            else:
                return 0.4
        
        return 0.5
        
    except Exception as e:
        print(f"Error in Fibonacci spiral analysis: {e}")
        return 0.5

def _calculate_visual_balance_advanced(self, objects_data: Dict) -> float:
    """
    Расчет визуального баланса на основе распределения визуальной массы.
    Учитывает цвет, размер, контрастность и позицию элементов.
    """
    try:
        if self.image_rgb is None:
            return 0.5
        
        h, w, _ = self.image_rgb.shape
        center_x, center_y = w // 2, h // 2
        
        # Создаем карту визуальных весов
        visual_weight_map = np.zeros((h, w), dtype=np.float32)
        
        # 1. Вес на основе яркости (контрастные области привлекают внимание)
        gray_normalized = self.image_gray.astype(np.float32) / 255.0
        brightness_weight = np.abs(gray_normalized - 0.5) * 2  # Отклонение от среднего
        visual_weight_map += brightness_weight * 0.3
        
        # 2. Вес на основе насыщенности цвета
        if self.image_hsv is not None:
            saturation = self.image_hsv[:, :, 1].astype(np.float32) / 255.0
            visual_weight_map += saturation * 0.2
        
        # 3. Вес на основе детектированных объектов
        if objects_data.get('high_confidence_objects'):
            for obj in objects_data['high_confidence_objects']:
                bbox = obj['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Ограничиваем координаты размерами изображения
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    # Добавляем вес объекта пропорционально его уверенности
                    weight_value = obj['confidence'] * 0.5
                    visual_weight_map[y1:y2, x1:x2] += weight_value
        
        # 4. Расчет моментов относительно центра
        # Создаем координатные сетки
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Смещения от центра
        x_offset = x_coords - center_x
        y_offset = y_coords - center_y
        
        # Моменты первого порядка (центр масс)
        total_weight = np.sum(visual_weight_map)
        
        if total_weight > 0:
            center_of_mass_x = np.sum(visual_weight_map * x_coords) / total_weight
            center_of_mass_y = np.sum(visual_weight_map * y_coords) / total_weight
            
            # Расстояние центра масс от геометрического центра
            com_distance = np.sqrt((center_of_mass_x - center_x)**2 + 
                                 (center_of_mass_y - center_y)**2)
            
            # Нормализуем расстояние
            max_distance = np.sqrt(w**2 + h**2) / 2
            normalized_distance = com_distance / max_distance
            
            # Чем ближе центр масс к геометрическому центру, тем лучше баланс
            balance_score = 1.0 - normalized_distance
            
        else:
            balance_score = 0.5
        
        # 5. Анализ квадрантов
        # Разделяем изображение на 4 квадранта и анализируем распределение весов
        quad_weights = []
        
        quad_weights.append(np.sum(visual_weight_map[:h//2, :w//2]))  # Верхний левый
        quad_weights.append(np.sum(visual_weight_map[:h//2, w//2:]))  # Верхний правый
        quad_weights.append(np.sum(visual_weight_map[h//2:, :w//2]))  # Нижний левый
        quad_weights.append(np.sum(visual_weight_map[h//2:, w//2:]))  # Нижний правый
        
        if sum(quad_weights) > 0:
            # Нормализуем веса квадрантов
            quad_weights = np.array(quad_weights) / sum(quad_weights)
            
            # Идеальный баланс - равномерное распределение (25% в каждом квадранте)
            ideal_distribution = np.array([0.25, 0.25, 0.25, 0.25])
            
            # Рассчитываем отклонение от идеального распределения
            distribution_diff = np.sum(np.abs(quad_weights - ideal_distribution))
            
            # Преобразуем в оценку качества (меньше отклонение = лучше баланс)
            quadrant_balance = 1.0 - (distribution_diff / 2.0)  # Максимальное отклонение = 2.0
            
        else:
            quadrant_balance = 0.5
        
        # 6. Комбинированная оценка
        final_balance = (balance_score * 0.6 + quadrant_balance * 0.4)
        
        return np.clip(final_balance, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in visual balance calculation: {e}")
        return 0.65

def _analyze_depth_cues_advanced(self) -> float:
    """
    Анализ визуальных подсказок глубины: перспектива, размеры, резкость.
    Основано на принципах восприятия трехмерного пространства.
    """
    try:
        if self.image_gray is None:
            return 0.5
        
        depth_score = 0.0
        
        # 1. Анализ резкости (фокусировка создает глубину)
        if CV2_AVAILABLE:
            # Вычисляем карту резкости с использованием лапласиана
            laplacian = cv2.Laplacian(self.image_gray, cv2.CV_64F)
            sharpness_map = np.abs(laplacian)
            
            # Анализируем распределение резкости
            h, w = sharpness_map.shape
            
            # Разделяем на зоны для анализа градиента резкости
            zones = [
                sharpness_map[:h//3, :],      # Верхняя треть (обычно фон)
                sharpness_map[h//3:2*h//3, :],  # Средняя треть (основной объект)
                sharpness_map[2*h//3:, :]     # Нижняя треть (передний план)
            ]
            
            zone_sharpness = [np.mean(zone) for zone in zones]
            
            # Проверяем наличие градиента резкости
            sharpness_variance = np.var(zone_sharpness)
            if sharpness_variance > 100:  # Есть значительное различие в резкости
                depth_score += 0.3
            
            # Анализ фокальной точки
            max_sharpness_idx = np.argmax(zone_sharpness)
            if max_sharpness_idx == 1:  # Максимальная резкость в центре
                depth_score += 0.2
        
        # 2. Анализ перспективных линий
        perspective_score = self._analyze_perspective_lines()
        depth_score += perspective_score * 0.3
        
        # 3. Анализ размерной перспективы (объекты разного размера)
        size_perspective_score = self._analyze_size_perspective()
        depth_score += size_perspective_score * 0.2
        
        # 4. Цветовая перспектива (теплые цвета кажутся ближе)
        color_depth_score = self._analyze_color_depth()
        depth_score += color_depth_score * 0.2
        
        return np.clip(depth_score, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in depth analysis: {e}")
        return 0.45

def _analyze_perspective_lines(self) -> float:
    """Анализ перспективных линий, сходящихся к точке схода."""
    try:
        if not CV2_AVAILABLE or self.image_gray is None:
            return 0.5
        
        # Детекция линий
        edges = cv2.Canny(self.image_gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
        
        if lines is None or len(lines) < 3:
            return 0.3
        
        h, w = self.image_gray.shape
        
        # Поиск точек пересечения линий
        intersection_points = []
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]
                
                # Вычисляем точку пересечения
                A = np.array([[np.cos(theta1), np.sin(theta1)],
                             [np.cos(theta2), np.sin(theta2)]])
                b = np.array([rho1, rho2])
                
                try:
                    intersection = np.linalg.solve(A, b)
                    x, y = intersection
                    
                    # Проверяем, что точка в разумных пределах
                    if -w <= x <= 2*w and -h <= y <= 2*h:
                        intersection_points.append((x, y))
                except np.linalg.LinAlgError:
                    continue
        
        if not intersection_points:
            return 0.4
        
        # Анализируем кластеризацию точек схода
        intersection_points = np.array(intersection_points)
        
        # Находим области концентрации точек
        from sklearn.cluster import DBSCAN
        
        try:
            clustering = DBSCAN(eps=min(w, h) * 0.1, min_samples=2).fit(intersection_points)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            
            if n_clusters >= 1:  # Есть хотя бы одна точка схода
                perspective_score = min(n_clusters * 0.3, 1.0)
                
                # Бонус если точка схода находится в интересном месте
                for label in set(clustering.labels_):
                    if label != -1:
                        cluster_points = intersection_points[clustering.labels_ == label]
                        cluster_center = np.mean(cluster_points, axis=0)
                        cx, cy = cluster_center
                        
                        # Точка схода в верхней части изображения (классическая перспектива)
                        if 0.2*w <= cx <= 0.8*w and 0 <= cy <= 0.4*h:
                            perspective_score += 0.2
                
                return min(perspective_score, 1.0)
            
        except ImportError:
            # Fallback без sklearn
            # Простой анализ разброса точек
            if len(intersection_points) > 1:
                distances = []
                for i in range(len(intersection_points)):
                    for j in range(i + 1, len(intersection_points)):
                        dist = np.linalg.norm(intersection_points[i] - intersection_points[j])
                        distances.append(dist)
                
                avg_distance = np.mean(distances)
                max_possible_dist = np.sqrt(w**2 + h**2)
                
                # Если точки близко друг к другу, возможно есть точка схода
                if avg_distance < max_possible_dist * 0.3:
                    return 0.7
                else:
                    return 0.4
        
        return 0.3
        
    except Exception as e:
        print(f"Error in perspective analysis: {e}")
        return 0.4

def _analyze_size_perspective(self) -> float:
    """Анализ размерной перспективы через различия в размерах объектов."""
    try:
        if not CV2_AVAILABLE or self.image_gray is None:
            return 0.5
        
        # Находим контуры объектов
        edges = cv2.Canny(self.image_gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:
            return 0.3
        
        # Анализируем размеры контуров
        areas = []
        centers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Игнорируем слишком маленькие объекты
                areas.append(area)
                
                # Находим центр контура
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
        
        if len(areas) < 2:
            return 0.3
        
        h, w = self.image_gray.shape
        
        # Анализируем зависимость размера от вертикальной позиции
        # (объекты ниже должны быть больше для создания перспективы)
        size_position_correlation = 0
        
        for i, (area, (cx, cy)) in enumerate(zip(areas, centers)):
            # Нормализуем позицию (0 = верх, 1 = низ)
            normalized_y = cy / h
            # Нормализуем размер
            normalized_area = area / max(areas)
            
            # Проверяем корреляцию: больше размер = ниже позиция
            if normalized_y > 0.5 and normalized_area > 0.5:  # Большие объекты внизу
                size_position_correlation += 0.2
            elif normalized_y < 0.5 and normalized_area < 0.5:  # Маленькие объекты вверху
                size_position_correlation += 0.2
        
        # Анализируем разнообразие размеров
        area_variance = np.var(areas) / (np.mean(areas)**2) if np.mean(areas) > 0 else 0
        
        size_diversity_score = min(area_variance, 1.0)
        
        # Комбинированная оценка
        perspective_score = (size_position_correlation + size_diversity_score * 0.5) / 1.5
        
        return np.clip(perspective_score, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in size perspective analysis: {e}")
        return 0.4

def _analyze_color_depth(self) -> float:
    """Анализ цветовой перспективы (теплые цвета ближе, холодные дальше)."""
    try:
        if self.image_rgb is None:
            return 0.5
        
        h, w, _ = self.image_rgb.shape
        
        # Анализируем цветовую температуру по зонам
        zones = [
            self.image_rgb[:h//3, :],      # Верхняя треть (фон)
            self.image_rgb[h//3:2*h//3, :],  # Средняя треть
            self.image_rgb[2*h//3:, :]     # Нижняя треть (передний план)
        ]
        
        zone_temperatures = []
        
        for zone in zones:
            # Вычисляем среднюю цветовую температуру зоны
            avg_red = np.mean(zone[:, :, 0])
            avg_blue = np.mean(zone[:, :, 2])
            
            # Теплота = отношение красного к синему
            if avg_blue > 0:
                temperature = avg_red / avg_blue
            else:
                temperature = avg_red / 1.0
            
            zone_temperatures.append(temperature)
        
        # Идеальная цветовая перспектива: теплее снизу, холоднее сверху
        temp_gradient = zone_temperatures[2] - zone_temperatures[0]  # Низ - Верх
        
        # Нормализуем градиент
        if temp_gradient > 0.1:  # Заметный теплый градиент снизу вверх
            gradient_score = min(temp_gradient / 2.0, 1.0)
        elif temp_gradient < -0.1:  # Обратный градиент (тоже может быть эффективным)
            gradient_score = min(abs(temp_gradient) / 3.0, 0.7)
        else:
            gradient_score = 0.3
        
        return np.clip(gradient_score, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in color depth analysis: {e}")
        return 0.4

# Третья очередь: Текстовые улучшения

def _calculate_text_readability_advanced(self, text_data: Dict) -> float:
    """
    Расчет читаемости на основе реальных OCR данных.
    Учитывает размер шрифта, контрастность, длину строк.
    """
    try:
        if not text_data.get('texts') or not text_data.get('positions'):
            return 1.0  # Если текста нет, читаемость идеальная
        
        readability_scores = []
        
        for i, (text, position) in enumerate(zip(text_data['texts'], text_data['positions'])):
            score = 0.0
            
            # 1. Оценка длины текста (очень длинные строки плохо читаются)
            text_length = len(text)
            if text_length <= 50:  # Оптимальная длина
                score += 0.3
            elif text_length <= 100:  # Приемлемая длина
                score += 0.2
            else:  # Слишком длинный текст
                score += 0.1
            
            # 2. Оценка размера на основе высоты bounding box
            text_height = position.get('height', 20)
            if text_height >= 24:  # Крупный, хорошо читаемый текст
                score += 0.3
            elif text_height >= 16:  # Средний размер
                score += 0.2
            elif text_height >= 12:  # Мелкий, но читаемый
                score += 0.15
            else:  # Слишком мелкий
                score += 0.05
            
            # 3. Уверенность OCR как показатель качества
            confidence = position.get('confidence', 0.5)
            score += confidence * 0.2
            
            # 4. Анализ символов (наличие специальных символов может снижать читаемость)
            clean_text = ''.join(c for c in text if c.isalnum() or c.isspace())
            char_ratio = len(clean_text) / len(text) if len(text) > 0 else 1.0
            score += char_ratio * 0.2
            
            readability_scores.append(min(score, 1.0))
        
        # Средняя читаемость по всем текстовым блокам
        avg_readability = np.mean(readability_scores) if readability_scores else 1.0
        
        return np.clip(avg_readability, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in text readability calculation: {e}")
        return 0.8

def _analyze_text_hierarchy_advanced(self, text_data: Dict) -> float:
    """
    Анализ иерархии текста на основе размеров и позиций блоков.
    """
    try:
        if not text_data.get('texts') or not text_data.get('positions'):
            return 1.0
        
        if len(text_data['texts']) <= 1:
            return 1.0  # Один блок текста не требует иерархии
        
        # Извлекаем размеры текстовых блоков
        text_heights = []
        text_areas = []
        y_positions = []
        
        for position in text_data['positions']:
            height = position.get('height', 20)
            width = position.get('width', 100)
            y_pos = position.get('y', 0)
            
            text_heights.append(height)
            text_areas.append(height * width)
            y_positions.append(y_pos)
        
        hierarchy_score = 0.0
        
        # 1. Анализ размерного разнообразия
        unique_heights = len(set([round(h/5)*5 for h in text_heights]))  # Группируем похожие размеры
        
        if unique_heights >= 3:  # Хорошая иерархия
            hierarchy_score += 0.4
        elif unique_heights == 2:  # Базовая иерархия
            hierarchy_score += 0.3
        else:  # Нет иерархии
            hierarchy_score += 0.1
        
        # 2. Проверка логичности иерархии (большие элементы сверху)
        if len(text_heights) >= 2:
            # Сортируем по вертикальной позиции
            sorted_indices = np.argsort(y_positions)
            sorted_heights = [text_heights[i] for i in sorted_indices]
            
            # Проверяем тренд размеров сверху вниз
            height_correlation = 0
            for i in range(len(sorted_heights) - 1):
                if sorted_heights[i] >= sorted_heights[i + 1]:  # Размер уменьшается вниз
                    height_correlation += 1
            
            correlation_ratio = height_correlation / (len(sorted_heights) - 1)
            hierarchy_score += correlation_ratio * 0.3
        
        # 3. Анализ позиционной иерархии
        if y_positions:
            y_range = max(y_positions) - min(y_positions)
            if self.image_gray is not None:
                img_height = self.image_gray.shape[0]
                vertical_spread = y_range / img_height if img_height > 0 else 0
                
                # Хорошая иерархия использует вертикальное пространство
                if 0.3 <= vertical_spread <= 0.8:
                    hierarchy_score += 0.3
                elif 0.1 <= vertical_spread <= 1.0:
                    hierarchy_score += 0.2
        
        return np.clip(hierarchy_score, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in text hierarchy analysis: {e}")
        return 0.7

def _analyze_text_positioning_advanced(self, text_data: Dict) -> float:
    """
    Анализ позиционирования текста относительно композиционных принципов.
    """
    try:
        if not text_data.get('texts') or not text_data.get('positions'):
            return 1.0
        
        if self.image_gray is None:
            return 0.75
        
        h, w = self.image_gray.shape
        positioning_score = 0.0
        
        # Линии правила третей
        third_lines_x = [w/3, 2*w/3]
        third_lines_y = [h/3, 2*h/3]
        
        for position in text_data['positions']:
            text_x = position.get('x', 0)
            text_y = position.get('y', 0)
            text_width = position.get('width', 0)
            text_height = position.get('height', 0)
            
            # Центр текстового блока
            center_x = text_x + text_width / 2
            center_y = text_y + text_height / 2
            
            # 1. Близость к линиям третей
            min_dist_x = min([abs(center_x - line) for line in third_lines_x])
            min_dist_y = min([abs(center_y - line) for line in third_lines_y])
            
            # Нормализуем расстояния
            norm_dist_x = min_dist_x / w
            norm_dist_y = min_dist_y / h
            
            if norm_dist_x < 0.05 or norm_dist_y < 0.05:  # Очень близко к линии третей
                positioning_score += 0.3
            elif norm_dist_x < 0.1 or norm_dist_y < 0.1:  # Близко
                positioning_score += 0.2
            
            # 2. Избегание краев (текст не должен быть слишком близко к краям)
            edge_margin = 0.05  # 5% от размеров изображения
            
            if (text_x > w * edge_margin and 
                text_y > h * edge_margin and 
                text_x + text_width < w * (1 - edge_margin) and 
                text_y + text_height < h * (1 - edge_margin)):
                positioning_score += 0.2
            
            # 3. Анализ вертикального позиционирования
            # CTA и важный текст лучше размещать в нижней половине
            vertical_position = center_y / h
            
            # Бонус за размещение в "золотой зоне" (нижняя треть)
            if 0.6 <= vertical_position <= 0.9:
                positioning_score += 0.1
        
        # Нормализуем на количество текстовых блоков
        if text_data['positions']:
            positioning_score /= len(text_data['positions'])
        
        return np.clip(positioning_score, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in text positioning analysis: {e}")
        return 0.75

def _analyze_text_contrast_advanced(self, text_data: Dict) -> float:
    """
    Анализ контрастности текста относительно фона.
    """
    try:
        if not text_data.get('positions') or self.image_gray is None:
            return 1.0
        
        contrast_scores = []
        
        for position in text_data['positions']:
            x = max(0, position.get('x', 0))
            y = max(0, position.get('y', 0))
            width = position.get('width', 50)
            height = position.get('height', 20)
            
            # Ограничиваем размеры изображения
            h, w = self.image_gray.shape
            x2 = min(w, x + width)
            y2 = min(h, y + height)
            
            if x2 > x and y2 > y:
                # Извлекаем область текста
                text_region = self.image_gray[y:y2, x:x2]
                
                if text_region.size > 0:
                    # Анализ контрастности в области текста
                    region_std = np.std(text_region)
                    region_range = np.ptp(text_region)  # Peak-to-peak (max - min)
                    
                    # Нормализуем контрастность
                    contrast_std = region_std / 128.0  # Половина от максимального диапазона
                    contrast_range = region_range / 255.0
                    
                    # Комбинированная оценка контрастности
                    combined_contrast = (contrast_std * 0.6 + contrast_range * 0.4)
                    
                    # Хорошая контрастность - от 0.3 до 1.0
                    if combined_contrast >= 0.3:
                        contrast_score = min(combined_contrast, 1.0)
                    else:
                        contrast_score = combined_contrast / 0.3 * 0.5  # Штраф за низкую контрастность
                    
                    contrast_scores.append(contrast_score)
        
        if contrast_scores:
            avg_contrast = np.mean(contrast_scores)
            return np.clip(avg_contrast, 0.0, 1.0)
        else:
            return 1.0
        
    except Exception as e:
        print(f"Error in text contrast analysis: {e}")
        return 0.85

# Дополнительные методы

def _analyze_center_focus_advanced(self, objects_data: Dict) -> float:
    """
    Анализ центрального фокуса с учетом детектированных объектов.
    """
    try:
        if self.image_gray is None:
            return 0.6
        
        h, w = self.image_gray.shape
        center_x, center_y = w // 2, h // 2
        
        focus_score = 0.0
        
        # 1. Анализ активности в центральной области
        center_radius = min(w, h) // 4
        y1 = max(0, center_y - center_radius)
        y2 = min(h, center_y + center_radius)
        x1 = max(0, center_x - center_radius)
        x2 = min(w, center_x + center_radius)
        
        if CV2_AVAILABLE:
            edges = cv2.Canny(self.image_gray, 50, 150)
            center_region = edges[y1:y2, x1:x2]
            total_region = edges
            
            center_activity = np.sum(center_region) / center_region.size if center_region.size > 0 else 0
            total_activity = np.sum(total_region) / total_region.size if total_region.size > 0 else 1
            
            activity_ratio = center_activity / total_activity if total_activity > 0 else 0
            
            if activity_ratio > 1.5:  # Центр более активен чем среднее
                focus_score += 0.4
            elif activity_ratio > 1.0:
                focus_score += 0.3
        
        # 2. Анализ объектов в центральной области
        if objects_data.get('high_confidence_objects'):
            objects_in_center = 0
            total_objects = len(objects_data['high_confidence_objects'])
            
            for obj in objects_data['high_confidence_objects']:
                obj_center = obj.get('center', [0, 0])
                if len(obj_center) >= 2:
                    obj_x, obj_y = obj_center[0], obj_center[1]
                    
                    # Проверяем, находится ли объект в центральной области
                    if (abs(obj_x - center_x) < center_radius and 
                        abs(obj_y - center_y) < center_radius):
                        objects_in_center += 1
            
            if total_objects > 0:
                center_object_ratio = objects_in_center / total_objects
                focus_score += center_object_ratio * 0.4
        
        # 3. Анализ яркостного распределения
        # Центр должен отличаться по яркости от периферии
        center_brightness = np.mean(self.image_gray[y1:y2, x1:x2])
        
        # Периферийные области
        border_width = min(w, h) // 10
        border_regions = [
            self.image_gray[:border_width, :],  # Верх
            self.image_gray[-border_width:, :],  # Низ
            self.image_gray[:, :border_width],  # Лево
            self.image_gray[:, -border_width:]  # Право
        ]
        
        border_brightness = np.mean([np.mean(region) for region in border_regions])
        
        brightness_diff = abs(center_brightness - border_brightness) / 255.0
        
        if brightness_diff > 0.1:  # Заметное различие
            focus_score += 0.2
        
        return np.clip(focus_score, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in center focus analysis: {e}")
        return 0.6

def _analyze_visual_flow(self, objects_data: Dict) -> float:
    """
    Анализ визуального потока - направления движения взгляда по изображению.
    """
    try:
        if self.image_gray is None:
            return 0.6
        
        flow_score = 0.0
        h, w = self.image_gray.shape
        
        # 1. Анализ расположения объектов для создания визуального пути
        if objects_data.get('high_confidence_objects'):
            objects = objects_data['high_confidence_objects']
            
            if len(objects) >= 2:
                # Сортируем объекты по позиции для анализа потока
                object_centers = []
                for obj in objects:
                    center = obj.get('center', [0, 0])
                    if len(center) >= 2:
                        object_centers.append((center[0], center[1]))
                
                if len(object_centers) >= 2:
                    # Анализ Z-паттерна (слева направо, сверху вниз)
                    z_pattern_score = self._analyze_z_pattern(object_centers, w, h)
                    flow_score += z_pattern_score * 0.4
                    
                    # Анализ F-паттерна (для текстового контента)
                    f_pattern_score = self._analyze_f_pattern(object_centers, w, h)
                    flow_score += f_pattern_score * 0.3
        
        # 2. Анализ направляющих элементов
        if CV2_AVAILABLE:
            # Поиск диагональных элементов, которые направляют взгляд
            edges = cv2.Canny(self.image_gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                diagonal_lines = 0
                for line in lines:
                    rho, theta = line[0]
                    angle_deg = np.degrees(theta)
                    
                    # Диагональные линии (30-60° и 120-150°) создают динамику
                    if (30 <= angle_deg <= 60) or (120 <= angle_deg <= 150):
                        diagonal_lines += 1
                
                if diagonal_lines > 0:
                    flow_score += min(diagonal_lines * 0.1, 0.3)
        
        return np.clip(flow_score, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in visual flow analysis: {e}")
        return 0.6

def _analyze_z_pattern(self, object_centers: List[Tuple[float, float]], w: int, h: int) -> float:
    """Анализ соответствия Z-паттерну чтения."""
    try:
        if len(object_centers) < 3:
            return 0.5
        
        # Сортируем объекты для анализа Z-паттерна
        # Z-паттерн: верхний левый -> верхний правый -> нижний левый -> нижний правый
        
        # Разделяем на верхнюю и нижнюю половины
        upper_objects = [(x, y) for x, y in object_centers if y < h/2]
        lower_objects = [(x, y) for x, y in object_centers if y >= h/2]
        
        z_score = 0.0
        
        if upper_objects and lower_objects:
            # В верхней половине: от левого к правому
            upper_sorted = sorted(upper_objects, key=lambda p: p[0])
            if len(upper_sorted) >= 2:
                leftmost_upper = upper_sorted[0]
                rightmost_upper = upper_sorted[-1]
                
                if rightmost_upper[0] > leftmost_upper[0]:  # Есть горизонтальный поток
                    z_score += 0.3
            
            # В нижней половине: желательно слева направо
            lower_sorted = sorted(lower_objects, key=lambda p: p[0])
            if len(lower_sorted) >= 2:
                leftmost_lower = lower_sorted[0]
                rightmost_lower = lower_sorted[-1]
                
                if rightmost_lower[0] > leftmost_lower[0]:
                    z_score += 0.3
            
            # Диагональная связь: правый верх к левому низу
            if upper_objects and lower_objects:
                rightmost_upper = max(upper_objects, key=lambda p: p[0])
                leftmost_lower = min(lower_objects, key=lambda p: p[0])
                
                # Проверяем диагональную связь
                if rightmost_upper[0] > leftmost_lower[0]:
                    z_score += 0.4
        
        return min(z_score, 1.0)
        
    except Exception as e:
        print(f"Error in Z-pattern analysis: {e}")
        return 0.5

def _analyze_f_pattern(self, object_centers: List[Tuple[float, float]], w: int, h: int) -> float:
    """Анализ соответствия F-паттерну чтения."""
    try:
        if len(object_centers) < 2:
            return 0.5
        
        # F-паттерн: горизонтальные движения в верхней части, вертикальное сканирование слева
        
        # Разделяем на верхнюю, среднюю и нижнюю трети
        upper_third = [(x, y) for x, y in object_centers if y < h/3]
        middle_third = [(x, y) for x, y in object_centers if h/3 <= y < 2*h/3]
        lower_third = [(x, y) for x, y in object_centers if y >= 2*h/3]
        
        f_score = 0.0
        
        # Горизонтальное сканирование в верхней части
        if upper_third and len(upper_third) >= 2:
            left_objects = [p for p in upper_third if p[0] < w/2]
            right_objects = [p for p in upper_third if p[0] >= w/2]
            
            if left_objects and right_objects:
                f_score += 0.3
        
        # Горизонтальное сканирование в средней части (короче чем в верхней)
        if middle_third:
            middle_span = max([p[0] for p in middle_third]) - min([p[0] for p in middle_third])
            upper_span = max([p[0] for p in upper_third]) - min([p[0] for p in upper_third]) if upper_third else 0
            
            if middle_span > 0 and upper_span > 0 and middle_span < upper_span:
                f_score += 0.2
        
        # Вертикальное сканирование слева
        left_side_objects = [(x, y) for x, y in object_centers if x < w/3]
        
        if len(left_side_objects) >= 2:
            # Проверяем вертикальное распределение
            y_positions = [p[1] for p in left_side_objects]
            y_range = max(y_positions) - min(y_positions)
            
            if y_range > h/3:  # Значительное вертикальное распределение
                f_score += 0.5
        
        return min(f_score, 1.0)
        
    except Exception as e:
        print(f"Error in F-pattern analysis: {e}")
        return 0.5

def _analyze_composition_dynamics(self, objects_data: Dict) -> float:
    """
    Анализ динамики композиции - движения, направления, энергии.
    """
    try:
        if self.image_gray is None:
            return 0.5
        
        dynamics_score = 0.0
        
        # 1. Анализ диагональных элементов (создают динамику)
        if CV2_AVAILABLE:
            edges = cv2.Canny(self.image_gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
            
            if lines is not None:
                diagonal_count = 0
                vertical_horizontal_count = 0
                
                for line in lines:
                    rho, theta = line[0]
                    angle_deg = np.degrees(theta)
                    
                    # Диагональные линии
                    if 30 <= angle_deg <= 60 or 120 <= angle_deg <= 150:
                        diagonal_count += 1
                    # Вертикальные и горизонтальные
                    elif (angle_deg <= 15 or angle_deg >= 165) or (75 <= angle_deg <= 105):
                        vertical_horizontal_count += 1
                
                total_lines = diagonal_count + vertical_horizontal_count
                if total_lines > 0:
                    diagonal_ratio = diagonal_count / total_lines
                    
                    # Больше диагоналей = больше динамики
                    if diagonal_ratio > 0.5:
                        dynamics_score += 0.4
                    elif diagonal_ratio > 0.3:
                        dynamics_score += 0.3
        
        # 2. Анализ направления объектов
        if objects_data.get('high_confidence_objects'):
            objects = objects_data['high_confidence_objects']
            
            if len(objects) >= 2:
                # Вычисляем векторы между объектами
                object_centers = []
                for obj in objects:
                    center = obj.get('center', [0, 0])
                    if len(center) >= 2:
                        object_centers.append(center)
                
                if len(object_centers) >= 2:
                    vectors = []
                    for i in range(len(object_centers) - 1):
                        dx = object_centers[i+1][0] - object_centers[i][0]
                        dy = object_centers[i+1][1] - object_centers[i][1]
                        vectors.append((dx, dy))
                    
                    # Анализ направленности векторов
                    if vectors:
                        # Вычисляем углы векторов
                        angles = [np.arctan2(dy, dx) for dx, dy in vectors]
                        
                        # Проверяем согласованность направлений
                        angle_variance = np.var(angles)
                        
                        # Умеренная вариация указывает на динамичную, но структурированную композицию
                        if 0.5 <= angle_variance <= 2.0:
                            dynamics_score += 0.3
                        elif angle_variance > 2.0:  # Высокая динамика
                            dynamics_score += 0.4
        
        # 3. Анализ асимметрии (создает напряжение и динамику)
        symmetry_score = self._calculate_symmetry_advanced()
        asymmetry_bonus = (1.0 - symmetry_score) * 0.3
        dynamics_score += asymmetry_bonus
        
        return np.clip(dynamics_score, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in composition dynamics analysis: {e}")
        return 0.5

def _calculate_text_to_image_ratio(self, text_data: Dict) -> float:
    """
    Расчет соотношения текста к изображению на основе площадей.
    """
    try:
        if not text_data.get('positions') or self.image_gray is None:
            return 0.0
        
        h, w = self.image_gray.shape
        total_image_area = h * w
        
        total_text_area = 0
        for position in text_data['positions']:
            text_width = position.get('width', 0)
            text_height = position.get('height', 0)
            total_text_area += text_width * text_height
        
        ratio = total_text_area / total_image_area if total_image_area > 0 else 0
        return np.clip(ratio, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in text-to-image ratio calculation: {e}")
        return 0.1

def _analyze_font_variety(self, text_data: Dict) -> int:
    """
    Анализ разнообразия шрифтов на основе размеров текстовых блоков.
    """
    try:
        if not text_data.get('positions'):
            return 0
        
        # Группируем размеры шрифтов
        heights = [position.get('height', 20) for position in text_data['positions']]
        
        # Округляем высоты для группировки похожих размеров
        rounded_heights = [round(h / 5) * 5 for h in heights]  # Группируем с шагом 5px
        
        unique_sizes = len(set(rounded_heights))
        return min(unique_sizes, 5)  # Максимум 5 различных размеров
        
    except Exception as e:
        print(f"Error in font variety analysis: {e}")
        return 2

def _calculate_text_density(self, text_data: Dict) -> float:
    """
    Расчет плотности текста (символов на единицу площади).
    """
    try:
        if not text_data.get('texts') or not text_data.get('positions'):
            return 0.0
        
        total_chars = sum(len(text) for text in text_data['texts'])
        total_area = 0
        
        for position in text_data['positions']:
            width = position.get('width', 0)
            height = position.get('height', 0)
            total_area += width * height
        
        if total_area > 0:
            density = total_chars / total_area
            # Нормализуем плотность (примерно 0.1 символа на квадратный пиксель = высокая плотность)
            normalized_density = min(density / 0.1, 1.0)
            return normalized_density
        
        return 0.0
        
    except Exception as e:
        print(f"Error in text density calculation: {e}")
        return 0.15

def _calculate_text_coverage(self, text_data: Dict) -> float:
    """
    Расчет покрытия изображения текстом.
    """
    try:
        if not text_data.get('positions') or self.image_gray is None:
            return 0.0
        
        h, w = self.image_gray.shape
        
        # Создаем маску покрытия текстом
        coverage_mask = np.zeros((h, w), dtype=bool)
        
        for position in text_data['positions']:
            x = max(0, position.get('x', 0))
            y = max(0, position.get('y', 0))
            width = position.get('width', 0)
            height = position.get('height', 0)
            
            x2 = min(w, x + width)
            y2 = min(h, y + height)
            
            if x2 > x and y2 > y:
                coverage_mask[y:y2, x:x2] = True
        
        coverage_ratio = np.sum(coverage_mask) / coverage_mask.size
        return np.clip(coverage_ratio, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in text coverage calculation: {e}")
        return 0.2

def _calculate_composition_complexity_advanced(self) -> float:
    """
    Улучшенный расчет сложности композиции.
    """
    try:
        if not CV2_AVAILABLE or self.image_gray is None:
            return 0.5
        
        # 1. Анализ количества краев
        edges = cv2.Canny(self.image_gray, 50, 150)
        edge_density = np.sum(edges) / (edges.size * 255)
        
        # 2. Анализ количества контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_complexity = len(contours) / 100.0  # Нормализация
        
        # 3. Анализ цветового разнообразия
        color_complexity = 0
        if self.image_rgb is not None:
            # Уменьшаем изображение для анализа
            small_img = cv2.resize(self.image_rgb, (50, 50))
            unique_colors = len(np.unique(small_img.reshape(-1, 3), axis=0))
            color_complexity = unique_colors / 1000.0  # Нормализация
        
        # 4. Текстурная сложность через локальные бинарные паттерны
        texture_complexity = 0
        try:
            # Простой анализ текстуры через дисперсию
            texture_complexity = np.var(self.image_gray) / (255**2)
        except:
            texture_complexity = 0.5
        
        # Комбинированная сложность
        total_complexity = (
            edge_density * 0.3 +
            contour_complexity * 0.25 +
            color_complexity * 0.25 +
            texture_complexity * 0.2
        )
        
        return np.clip(total_complexity, 0.0, 1.0)
        
    except Exception as e:
        print(f"Error in composition complexity calculation: {e}")
        return 0.5

# Алиас для обратной совместимости
ImageAnalyzer = AdvancedImageAnalyzer
