# image_analyzer.py
"""
Модуль анализа изображений для Creative Performance Predictor.
Извлекает количественные характеристики из креативных материалов.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
from collections import Counter
import pytesseract
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import COLOR_ANALYSIS, COMPOSITION_ANALYSIS, TEXT_ANALYSIS, get_color_name

class ImageAnalyzer:
    """
    Класс для комплексного анализа изображений креативов.
    Извлекает цветовые, композиционные и текстовые характеристики.
    """
    
    def __init__(self):
        self.image = None
        self.image_rgb = None
        self.image_hsv = None
        self.image_gray = None
        self.features = {}
        
    def load_image(self, image_data) -> bool:
        """
        Загрузить изображение для анализа.
        
        Args:
            image_data: Данные изображения (PIL Image или numpy array)
            
        Returns:
            bool: Успешность загрузки
        """
        try:
            if isinstance(image_data, Image.Image):
                self.image = image_data
            else:
                self.image = Image.fromarray(image_data)
            
            # Конвертация в различные цветовые пространства
            self.image_rgb = np.array(self.image.convert('RGB'))
            self.image_hsv = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2HSV)
            self.image_gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
            
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return False
    
    def analyze_colors(self) -> Dict:
        """
        Анализ цветовых характеристик изображения.
        
        Returns:
            Dict: Словарь с результатами цветового анализа
        """
        if self.image_rgb is None:
            return {}
        
        # Получение доминирующих цветов
        dominant_colors = self._get_dominant_colors()
        
        # Анализ цветовой гармонии
        harmony_score = self._calculate_color_harmony(dominant_colors)
        
        # Анализ контрастности
        contrast_score = self._calculate_contrast()
        
        # Цветовая температура
        temperature = self._calculate_color_temperature()
        
        # Насыщенность и яркость
        saturation = np.mean(self.image_hsv[:, :, 1]) / 255.0
        brightness = np.mean(self.image_hsv[:, :, 2]) / 255.0
        
        return {
            'dominant_colors': dominant_colors,
            'harmony_score': harmony_score,
            'contrast_score': contrast_score,
            'color_temperature': temperature,
            'saturation': saturation,
            'brightness': brightness,
            'color_diversity': len(dominant_colors),
            'warm_cool_ratio': self._calculate_warm_cool_ratio(dominant_colors)
        }
    
    def analyze_composition(self) -> Dict:
        """
        Анализ композиционных характеристик изображения.
        
        Returns:
            Dict: Словарь с результатами композиционного анализа
        """
        if self.image_gray is None:
            return {}
        
        # Правило третей
        rule_of_thirds_score = self._analyze_rule_of_thirds()
        
        # Визуальный баланс
        balance_score = self._calculate_visual_balance()
        
        # Сложность композиции
        complexity_score = self._calculate_composition_complexity()
        
        # Центрирование объектов
        center_focus_score = self._analyze_center_focus()
        
        # Направляющие линии
        leading_lines_score = self._detect_leading_lines()
        
        return {
            'rule_of_thirds_score': rule_of_thirds_score,
            'visual_balance_score': balance_score,
            'composition_complexity': complexity_score,
            'center_focus_score': center_focus_score,
            'leading_lines_score': leading_lines_score,
            'symmetry_score': self._calculate_symmetry(),
            'depth_perception': self._analyze_depth_cues()
        }
    
    def analyze_text(self) -> Dict:
        """
        Анализ текстовых элементов в изображении.
        
        Returns:
            Dict: Словарь с результатами текстового анализа
        """
        if self.image_rgb is None:
            return {}
        
        try:
            # OCR для извлечения текста
            ocr_data = pytesseract.image_to_data(
                self.image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'
            )
            
            # Фильтрация надежного текста
            reliable_text = []
            text_positions = []
            font_sizes = []
            
            for i in range(len(ocr_data['text'])):
                confidence = int(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()
                
                if confidence > TEXT_ANALYSIS['min_text_confidence'] and text:
                    reliable_text.append(text)
                    text_positions.append({
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    })
                    font_sizes.append(ocr_data['height'][i])
            
            # Анализ характеристик текста
            text_analysis = {
                'text_amount': len(reliable_text),
                'total_characters': sum(len(text) for text in reliable_text),
                'readability_score': self._calculate_text_readability(text_positions),
                'text_hierarchy': self._analyze_text_hierarchy(font_sizes),
                'text_positioning': self._analyze_text_positioning(text_positions),
                'text_contrast': self._analyze_text_contrast(text_positions),
                'has_cta': self._detect_cta_elements(reliable_text)
            }
            
            return text_analysis
            
        except Exception as e:
            print(f"Ошибка анализа текста: {e}")
            return {
                'text_amount': 0,
                'total_characters': 0,
                'readability_score': 0.5,
                'text_hierarchy': 0.5,
                'text_positioning': 0.5,
                'text_contrast': 0.5,
                'has_cta': False
            }
    
    def get_all_features(self) -> Dict:
        """
        Получить все извлеченные признаки как единый вектор для ML.
        
        Returns:
            Dict: Словарь всех числовых признаков
        """
        color_features = self.analyze_colors()
        composition_features = self.analyze_composition()
        text_features = self.analyze_text()
        
        # Объединение всех признаков
        all_features = {
            # Цветовые признаки
            'brightness': color_features.get('brightness', 0),
            'saturation': color_features.get('saturation', 0),
            'contrast_score': color_features.get('contrast_score', 0),
            'color_temperature': color_features.get('color_temperature', 0),
            'harmony_score': color_features.get('harmony_score', 0),
            'color_diversity': color_features.get('color_diversity', 0),
            'warm_cool_ratio': color_features.get('warm_cool_ratio', 0),
            
            # Композиционные признаки
            'rule_of_thirds_score': composition_features.get('rule_of_thirds_score', 0),
            'visual_balance_score': composition_features.get('visual_balance_score', 0),
            'composition_complexity': composition_features.get('composition_complexity', 0),
            'center_focus_score': composition_features.get('center_focus_score', 0),
            'leading_lines_score': composition_features.get('leading_lines_score', 0),
            'symmetry_score': composition_features.get('symmetry_score', 0),
            'depth_perception': composition_features.get('depth_perception', 0),
            
            # Текстовые признаки
            'text_amount': text_features.get('text_amount', 0),
            'total_characters': text_features.get('total_characters', 0),
            'readability_score': text_features.get('readability_score', 0),
            'text_hierarchy': text_features.get('text_hierarchy', 0),
            'text_positioning': text_features.get('text_positioning', 0),
            'text_contrast': text_features.get('text_contrast', 0),
            'has_cta': int(text_features.get('has_cta', False)),
            
            # Дополнительные метрики
            'aspect_ratio': self._calculate_aspect_ratio(),
            'image_size_score': self._calculate_size_score()
        }
        
        self.features = all_features
        return all_features
    
    # === ПРИВАТНЫЕ МЕТОДЫ ===
    
    def _get_dominant_colors(self, n_colors: int = None) -> List[Tuple[int, int, int]]:
        """Получить доминирующие цвета изображения."""
        if n_colors is None:
            n_colors = COLOR_ANALYSIS['n_dominant_colors']
        
        # Reshape изображения для кластеризации
        pixels = self.image_rgb.reshape(-1, 3)
        
        # Уменьшение выборки для ускорения
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # K-means кластеризация
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Получение центров кластеров
        colors = kmeans.cluster_centers_.astype(int)
        
        # Подсчет частоты каждого цвета
        labels = kmeans.labels_
        label_counts = Counter(labels)
        
        # Сортировка по частоте
        dominant_colors = []
        for i in sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True):
            dominant_colors.append(tuple(colors[i]))
        
        return dominant_colors
    
    def _calculate_color_harmony(self, colors: List[Tuple]) -> float:
        """Рассчитать индекс цветовой гармонии."""
        if len(colors) < 2:
            return 0.5
        
        # Конвертация в HSV для анализа гармонии
        hsv_colors = []
        for color in colors:
            r, g, b = [x/255.0 for x in color]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsv_colors.append((h * 360, s, v))
        
        harmony_score = 0
        total_pairs = 0
        
        # Анализ пар цветов
        for i in range(len(hsv_colors)):
            for j in range(i+1, len(hsv_colors)):
                h1, s1, v1 = hsv_colors[i]
                h2, s2, v2 = hsv_colors[j]
                
                # Разность оттенков
                hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
                
                # Проверка на гармоничные соотношения
                if self._is_harmonic_interval(hue_diff):
                    harmony_score += 1
                
                total_pairs += 1
        
        return harmony_score / total_pairs if total_pairs > 0 else 0.5
    
    def _is_harmonic_interval(self, hue_diff: float) -> bool:
        """Проверить, является ли интервал оттенков гармоничным."""
        harmonic_intervals = [
            (0, 30),      # Монохромная
            (150, 210),   # Комплементарная
            (110, 130),   # Триадная
            (80, 100),    # Аналогичная
            (25, 35)      # Соседние
        ]
        
        for min_diff, max_diff in harmonic_intervals:
            if min_diff <= hue_diff <= max_diff:
                return True
        return False
    
    def _calculate_contrast(self) -> float:
        """Рассчитать общую контрастность изображения."""
        # Используем стандартное отклонение как меру контрастности
        std_dev = np.std(self.image_gray)
        # Нормализация к диапазону [0, 1]
        contrast = min(std_dev / 128.0, 1.0)
        return contrast
    
    def _calculate_color_temperature(self) -> float:
        """Рассчитать цветовую температуру (теплая/холодная)."""
        # Анализ красных/синих компонентов
        red_mean = np.mean(self.image_rgb[:, :, 0])
        blue_mean = np.mean(self.image_rgb[:, :, 2])
        
        # Нормализация температуры к диапазону [0, 1]
        # 0 = холодная, 1 = теплая
        if red_mean + blue_mean > 0:
            temperature = red_mean / (red_mean + blue_mean)
        else:
            temperature = 0.5
        
        return temperature
    
    def _calculate_warm_cool_ratio(self, colors: List[Tuple]) -> float:
        """Рассчитать соотношение теплых к холодным цветам."""
        warm_count = 0
        cool_count = 0
        
        for color in colors:
            r, g, b = color
            # Простая эвристика для определения теплоты цвета
            if r > b + 30:  # Красный больше синего
                warm_count += 1
            elif b > r + 30:  # Синий больше красного
                cool_count += 1
        
        total = warm_count + cool_count
        return warm_count / total if total > 0 else 0.5
    
    def _analyze_rule_of_thirds(self) -> float:
        """Анализ соответствия правилу третей."""
        height, width = self.image_gray.shape
        
        # Линии третей
        h_lines = [height // 3, 2 * height // 3]
        v_lines = [width // 3, 2 * width // 3]
        
        # Детекция краев для поиска объектов
        edges = cv2.Canny(self.image_gray, 50, 150)
        
        # Поиск контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.5
        
        # Анализ расположения основных объектов
        score = 0
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Только крупные объекты
                # Центр объекта
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Проверка близости к точкам пересечения третей
                    for h_line in h_lines:
                        for v_line in v_lines:
                            distance = np.sqrt((cx - v_line)**2 + (cy - h_line)**2)
                            if distance < min(width, height) * 0.1:  # 10% от размера
                                score += 1
        
        return min(score / len(contours), 1.0) if contours else 0.5
    
    def _calculate_visual_balance(self) -> float:
        """Рассчитать визуальный баланс изображения."""
        height, width = self.image_gray.shape
        
        # Разделение на левую и правую половины
        left_half = self.image_gray[:, :width//2]
        right_half = self.image_gray[:, width//2:]
        
        # Расчет визуального "веса" каждой половины
        left_weight = np.sum(255 - left_half)  # Инвертируем для веса
        right_weight = np.sum(255 - right_half)
        
        # Баланс как соотношение весов
        total_weight = left_weight + right_weight
        if total_weight > 0:
            balance = min(left_weight, right_weight) / (total_weight / 2)
        else:
            balance = 1.0
        
        return balance
    
    def _calculate_composition_complexity(self) -> float:
        """Рассчитать сложность композиции."""
        # Детекция краев
        edges = cv2.Canny(self.image_gray, 30, 100)
        
        # Количество краев как мера сложности
        edge_density = np.sum(edges > 0) / edges.size
        
        # Нормализация к диапазону [0, 1]
        complexity = min(edge_density * 10, 1.0)
        
        return complexity
    
    def _analyze_center_focus(self) -> float:
        """Анализ центрирования основных объектов."""
        height, width = self.image_gray.shape
        center_x, center_y = width // 2, height // 2
        
        # Создание маски центральной области (30% от размера)
        mask_size = min(width, height) * 0.15
        center_region = self.image_gray[
            int(center_y - mask_size):int(center_y + mask_size),
            int(center_x - mask_size):int(center_x + mask_size)
        ]
        
        if center_region.size == 0:
            return 0.5
        
        # Анализ контрастности в центральной области
        center_contrast = np.std(center_region)
        overall_contrast = np.std(self.image_gray)
        
        # Соотношение контрастности как мера фокуса
        focus_score = center_contrast / overall_contrast if overall_contrast > 0 else 0.5
        
        return min(focus_score, 1.0)
    
    def _detect_leading_lines(self) -> float:
        """Детекция направляющих линий."""
        # Преобразование Хафа для поиска линий
        edges = cv2.Canny(self.image_gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.0
        
        # Анализ направления линий
        diagonal_lines = 0
        total_lines = len(lines)
        
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            
            # Проверка на диагональные линии (направляющие)
            if 30 <= angle <= 60 or 120 <= angle <= 150:
                diagonal_lines += 1
        
        return diagonal_lines / total_lines if total_lines > 0 else 0.0
    
    def _calculate_symmetry(self) -> float:
        """Рассчитать симметричность изображения."""
        height, width = self.image_gray.shape
        
        # Вертикальная симметрия
        left_half = self.image_gray[:, :width//2]
        right_half = np.fliplr(self.image_gray[:, width//2:])
        
        # Убедимся, что размеры совпадают
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Корреляция между половинами
        correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        
        # Обработка NaN значений
        if np.isnan(correlation):
            correlation = 0.0
        
        return max(0, correlation)
    
    def _analyze_depth_cues(self) -> float:
        """Анализ признаков глубины изображения."""
        # Анализ размытия как индикатора глубины
        laplacian_var = cv2.Laplacian(self.image_gray, cv2.CV_64F).var()
        
        # Нормализация вариации лапласиана
        depth_score = min(laplacian_var / 1000, 1.0)
        
        return depth_score
    
    def _calculate_text_readability(self, text_positions: List[Dict]) -> float:
        """Рассчитать читаемость текста."""
        if not text_positions:
            return 1.0  # Нет текста = максимальная "читаемость"
        
        readability_scores = []
        
        for pos in text_positions:
            # Извлечение области текста
            x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
            
            # Проверка границ
            if x < 0 or y < 0 or x + w > self.image_rgb.shape[1] or y + h > self.image_rgb.shape[0]:
                continue
            
            text_region = self.image_rgb[y:y+h, x:x+w]
            
            if text_region.size == 0:
                continue
            
            # Анализ контрастности в области текста
            gray_region = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray_region)
            
            # Нормализация контрастности
            readability = min(contrast / 128, 1.0)
            readability_scores.append(readability)
        
        return np.mean(readability_scores) if readability_scores else 0.5
    
    def _analyze_text_hierarchy(self, font_sizes: List[int]) -> float:
        """Анализ иерархии текстовых элементов."""
        if len(font_sizes) < 2:
            return 1.0 if len(font_sizes) == 1 else 0.5
        
        # Подсчет различных размеров шрифтов
        unique_sizes = len(set(font_sizes))
        
        # Хорошая иерархия имеет 2-4 различных размера
        if 2 <= unique_sizes <= 4:
            hierarchy_score = 1.0
        elif unique_sizes == 1:
            hierarchy_score = 0.3  # Нет иерархии
        else:
            hierarchy_score = 0.6  # Слишком много уровней
        
        return hierarchy_score
    
    def _analyze_text_positioning(self, text_positions: List[Dict]) -> float:
        """Анализ позиционирования текста."""
        if not text_positions:
            return 1.0
        
        height, width = self.image_gray.shape
        
        # Анализ распределения текста по областям
        top_area = sum(1 for pos in text_positions if pos['y'] < height * 0.3)
        middle_area = sum(1 for pos in text_positions if height * 0.3 <= pos['y'] < height * 0.7)
        bottom_area = sum(1 for pos in text_positions if pos['y'] >= height * 0.7)
        
        total_text = len(text_positions)
        
        # Хорошее позиционирование - текст не сконцентрирован в одной области
        distribution_score = 1.0 - max(top_area, middle_area, bottom_area) / total_text
        
        return distribution_score
    
    def _analyze_text_contrast(self, text_positions: List[Dict]) -> float:
        """Анализ контрастности текста относительно фона."""
        if not text_positions:
            return 1.0
        
        contrast_scores = []
        
        for pos in text_positions:
            x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
            
            # Проверка границ
            if x < 0 or y < 0 or x + w > self.image_rgb.shape[1] or y + h > self.image_rgb.shape[0]:
                continue
            
            # Область текста
            text_region = self.image_gray[y:y+h, x:x+w]
            
            if text_region.size == 0:
                continue
            
            # Расширенная область для анализа фона
            bg_margin = 10
            bg_x1 = max(0, x - bg_margin)
            bg_y1 = max(0, y - bg_margin)
            bg_x2 = min(self.image_gray.shape[1], x + w + bg_margin)
            bg_y2 = min(self.image_gray.shape[0], y + h + bg_margin)
            
            bg_region = self.image_gray[bg_y1:bg_y2, bg_x1:bg_x2]
            
            # Расчет контрастности
            text_brightness = np.mean(text_region)
            bg_brightness = np.mean(bg_region)
            
            contrast = abs(text_brightness - bg_brightness) / 255.0
            contrast_scores.append(contrast)
        
        return np.mean(contrast_scores) if contrast_scores else 0.5
    
    def _detect_cta_elements(self, texts: List[str]) -> bool:
        """Детекция элементов призыва к действию."""
        cta_keywords = [
            'купить', 'заказать', 'скачать', 'получить', 'узнать',
            'попробовать', 'зарегистрироваться', 'подписаться',
            'звонить', 'написать', 'перейти', 'кликнуть',
            'buy', 'order', 'download', 'get', 'try', 'subscribe'
        ]
        
        # Поиск CTA ключевых слов в тексте
        all_text = ' '.join(texts).lower()
        
        for keyword in cta_keywords:
            if keyword in all_text:
                return True
        
        return False
    
    def _calculate_aspect_ratio(self) -> float:
        """Рассчитать соотношение сторон изображения."""
        if self.image is None:
            return 1.0
        
        width, height = self.image.size
        return width / height
    
    def _calculate_size_score(self) -> float:
        """Рассчитать оценку размера изображения."""
        if self.image is None:
            return 0.5
        
        width, height = self.image.size
        total_pixels = width * height
        
        # Нормализация относительно стандартных размеров рекламы
        # 1920x1080 = 2,073,600 пикселей как базовый размер
        base_size = 1920 * 1080
        size_score = min(total_pixels / base_size, 2.0) / 2.0
        
        return size_score