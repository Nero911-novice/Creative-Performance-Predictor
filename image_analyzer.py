# image_analyzer.py
"""
Модуль анализа изображений для Creative Performance Predictor.
Извлекает количественные характеристики из креативных материалов.
"""

# Безопасные импорты с fallback вариантами
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
from collections import Counter
import re
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.signal import convolve2d
warnings.filterwarnings('ignore')

# Опциональные импорты с обработкой ошибок
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV не установлен. Некоторые функции будут недоступны или упрощены.")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: Tesseract OCR не установлен. Анализ текста будет упрощенным.")

from config import COLOR_ANALYSIS, TEXT_ANALYSIS, get_color_name

class ImageAnalyzer:
    """
    Класс для комплексного анализа изображений креативов.
    Извлекает цветовые, композиционные и текстовые характеристики.
    """
    
    def __init__(self):
        self.image: Optional[Image.Image] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.image_hsv: Optional[np.ndarray] = None
        self.image_gray: Optional[np.ndarray] = None
        self.features: Dict[str, Any] = {}
        
    def load_image(self, image_data) -> bool:
        """Загрузить изображение для анализа."""
        try:
            if isinstance(image_data, Image.Image):
                self.image = image_data.convert('RGB')
            else:
                self.image = Image.fromarray(image_data).convert('RGB')
            
            self.image_rgb = np.array(self.image)
            
            if CV2_AVAILABLE:
                self.image_hsv = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2HSV)
                self.image_gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
            else:
                self.image_hsv = self._rgb_to_hsv_numpy(self.image_rgb)
                self.image_gray = np.dot(self.image_rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            
            return True
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return False
    
    def analyze_colors(self) -> Dict:
        """Анализ цветовых характеристик изображения."""
        if self.image_rgb is None or self.image_hsv is None: return {}
        dominant_colors = self._get_dominant_colors()
        return {
            'dominant_colors': dominant_colors,
            'harmony_score': self._calculate_color_harmony(dominant_colors),
            'contrast_score': self._calculate_contrast(),
            'color_temperature': self._calculate_color_temperature(),
            'saturation': np.mean(self.image_hsv[:, :, 1]) / 255.0,
            'brightness': np.mean(self.image_hsv[:, :, 2]) / 255.0,
            'color_diversity': len(dominant_colors),
            'warm_cool_ratio': self._calculate_warm_cool_ratio(dominant_colors)
        }
    
    def analyze_composition(self) -> Dict:
        """Анализ композиционных характеристик изображения."""
        if self.image_gray is None: return {}
        return {
            'rule_of_thirds_score': self._analyze_rule_of_thirds(),
            'visual_balance_score': self._calculate_visual_balance(),
            'composition_complexity': self._calculate_composition_complexity(),
            'center_focus_score': self._analyze_center_focus(),
            'leading_lines_score': self._detect_leading_lines(),
            'symmetry_score': self._calculate_symmetry(),
            'depth_perception': self._analyze_depth_cues()
        }
    
    def analyze_text(self) -> Dict:
        """Анализ текстовых элементов в изображении."""
        if self.image_rgb is None: return {}
        
        if not TESSERACT_AVAILABLE:
            return {
                'text_amount': 2, 'total_characters': 50, 'readability_score': 0.7,
                'text_hierarchy': 0.6, 'text_positioning': 0.5, 'text_contrast': 0.6, 'has_cta': True
            }
        
        try:
            ocr_data = pytesseract.image_to_data(self.image, output_type=pytesseract.Output.DICT, config='--psm 6')
            
            reliable_text, text_positions, font_sizes = [], [], []
            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > TEXT_ANALYSIS['min_text_confidence'] and ocr_data['text'][i].strip():
                    reliable_text.append(ocr_data['text'][i])
                    text_positions.append({'x': ocr_data['left'][i], 'y': ocr_data['top'][i], 'width': ocr_data['width'][i], 'height': ocr_data['height'][i]})
                    font_sizes.append(ocr_data['height'][i])
            
            return {
                'text_amount': len(reliable_text), 'total_characters': sum(len(t) for t in reliable_text),
                'readability_score': self._calculate_text_readability(text_positions),
                'text_hierarchy': self._analyze_text_hierarchy(font_sizes),
                'text_positioning': self._analyze_text_positioning(text_positions),
                'text_contrast': self._analyze_text_contrast(text_positions),
                'has_cta': self._detect_cta_elements(reliable_text)
            }
        except Exception as e:
            print(f"Ошибка анализа текста: {e}")
            return {'text_amount': 0, 'total_characters': 0, 'readability_score': 0.5, 'text_hierarchy': 0.5, 'text_positioning': 0.5, 'text_contrast': 0.5, 'has_cta': False}

    def get_all_features(self) -> Dict:
        """Получить все извлеченные признаки как единый вектор для ML."""
        color_features = self.analyze_colors()
        composition_features = self.analyze_composition()
        text_features = self.analyze_text()
        
        all_features = {
            **{k: v for k, v in color_features.items() if not isinstance(v, list)},
            **composition_features,
            **text_features,
            'aspect_ratio': self._calculate_aspect_ratio(),
            'image_size_score': self._calculate_size_score()
        }
        all_features['has_cta'] = int(all_features.get('has_cta', False))
        
        self.features = all_features
        return all_features
    
    def _rgb_to_hsv_numpy(self, rgb_image):
        rgb_norm = rgb_image.astype(np.float32) / 255.0
        hsv = np.zeros_like(rgb_norm, dtype=np.float32)
        r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]
        max_c, min_c = np.max(rgb_norm, axis=2), np.min(rgb_norm, axis=2)
        delta = max_c - min_c
        
        hsv[..., 2] = max_c  # V
        s_mask = max_c != 0
        hsv[s_mask, 1] = delta[s_mask] / max_c[s_mask] # S
        
        h_mask = delta != 0
        idx_r = (max_c == r) & h_mask
        idx_g = (max_c == g) & h_mask
        idx_b = (max_c == b) & h_mask
        
        hsv[idx_r, 0] = 60 * (((g - b) / delta)[idx_r] % 6)
        hsv[idx_g, 0] = 60 * (((b - r) / delta)[idx_g] + 2)
        hsv[idx_b, 0] = 60 * (((r - g) / delta)[idx_b] + 4)
        
        hsv[..., 0] = hsv[..., 0] / 2 # Scale H to 0-179
        hsv[..., 1] = hsv[..., 1] * 255 # Scale S to 0-255
        hsv[..., 2] = hsv[..., 2] * 255 # Scale V to 0-255
        return hsv.astype(np.uint8)

    def _get_dominant_colors(self, n_colors: Optional[int] = None) -> List[Tuple[int, int, int]]:
        n_colors = n_colors or COLOR_ANALYSIS['n_dominant_colors']
        pixels = self.image_rgb.reshape(-1, 3)
        if len(pixels) > 10000:
            pixels = pixels[np.random.choice(len(pixels), 10000, replace=False)]
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto').fit(pixels)
        return [tuple(map(int, color)) for color in kmeans.cluster_centers_]

    def _calculate_color_harmony(self, colors: List[Tuple]) -> float:
        if len(colors) < 2: return 0.5
        hsv_colors = [colorsys.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255) for c in colors]
        harmony_score, total_pairs = 0, 0
        for i in range(len(hsv_colors)):
            for j in range(i + 1, len(hsv_colors)):
                hue_diff = abs(hsv_colors[i][0] - hsv_colors[j][0]) * 360
                hue_diff = min(hue_diff, 360 - hue_diff)
                if any(low <= hue_diff <= high for low, high in [(0, 30), (150, 210), (110, 130)]):
                    harmony_score += 1
                total_pairs += 1
        return harmony_score / total_pairs if total_pairs > 0 else 0.5

    def _calculate_contrast(self) -> float:
        return min(np.std(self.image_gray) / 128.0, 1.0)

    def _calculate_color_temperature(self) -> float:
        red_mean = np.mean(self.image_rgb[:, :, 0])
        blue_mean = np.mean(self.image_rgb[:, :, 2])
        return red_mean / (red_mean + blue_mean) if red_mean + blue_mean > 0 else 0.5

    def _calculate_warm_cool_ratio(self, colors: List[Tuple]) -> float:
        warm = sum(1 for r, g, b in colors if r > b + 30)
        cool = sum(1 for r, g, b in colors if b > r + 30)
        return warm / (warm + cool) if warm + cool > 0 else 0.5

    def _analyze_rule_of_thirds(self) -> float:
        if not CV2_AVAILABLE: return 0.5
        edges = cv2.Canny(self.image_gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0.0
        
        h, w = self.image_gray.shape
        h_lines, v_lines = [h//3, 2*h//3], [w//3, 2*w//3]
        score = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    if any(np.sqrt((cx-vl)**2 + (cy-hl)**2) < min(w,h)*0.1 for hl in h_lines for vl in v_lines):
                        score += 1
        return min(score / len(contours), 1.0) if contours else 0.0

    def _calculate_visual_balance(self) -> float:
        h, w = self.image_gray.shape
        left_half, right_half = self.image_gray[:, :w//2], self.image_gray[:, w//2:]
        left_weight, right_weight = np.sum(255 - left_half), np.sum(255 - right_half)
        total_weight = left_weight + right_weight
        return min(left_weight, right_weight) / (total_weight / 2) if total_weight > 0 else 1.0

    def _calculate_composition_complexity(self) -> float:
        if CV2_AVAILABLE:
            edges = cv2.Canny(self.image_gray, 30, 100)
        else:
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            grad_x = convolve2d(self.image_gray, kernel_x, mode='same', boundary='symm')
            grad_y = np.transpose(convolve2d(self.image_gray, np.transpose(kernel_x), mode='same', boundary='symm'))
            edges = np.sqrt(grad_x**2 + grad_y**2)
        return min(np.sum(edges > 50) / edges.size * 10, 1.0)

    def _analyze_center_focus(self) -> float:
        h, w = self.image_gray.shape
        center_y, center_x = h // 2, w // 2
        mask_size = int(min(w, h) * 0.15)
        center_region = self.image_gray[center_y - mask_size : center_y + mask_size, center_x - mask_size : center_x + mask_size]
        if center_region.size == 0: return 0.5
        center_contrast = np.std(center_region)
        overall_contrast = np.std(self.image_gray)
        return min(center_contrast / overall_contrast, 1.0) if overall_contrast > 0 else 0.5

    def _detect_leading_lines(self) -> float:
        if not CV2_AVAILABLE: return 0.3
        edges = cv2.Canny(self.image_gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None: return 0.0
        diagonal_lines = sum(1 for line in lines for rho, theta in line if 30 <= (theta*180/np.pi) <= 60 or 120 <= (theta*180/np.pi) <= 150)
        return diagonal_lines / len(lines)

    def _calculate_symmetry(self) -> float:
        h, w = self.image_gray.shape
        left, right = self.image_gray[:, :w//2], np.fliplr(self.image_gray[:, w//2:])
        min_w = min(left.shape[1], right.shape[1])
        correlation = np.corrcoef(left[:, :min_w].flatten(), right[:, :min_w].flatten())[0, 1]
        return max(0, correlation) if not np.isnan(correlation) else 0.0

    def _analyze_depth_cues(self) -> float:
        if CV2_AVAILABLE:
            laplacian_var = cv2.Laplacian(self.image_gray, cv2.CV_64F).var()
        else:
            laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            laplacian = convolve2d(self.image_gray.astype(np.float64), laplacian_kernel, mode='same')
            laplacian_var = np.var(laplacian)
        return min(laplacian_var / 1000, 1.0)

    def _calculate_text_readability(self, text_positions: List[Dict]) -> float:
        # ИСПРАВЛЕНИЕ: Добавлена проверка на CV2_AVAILABLE
        if not text_positions or not CV2_AVAILABLE:
            return 1.0 if not text_positions else 0.5
        readability_scores = []
        for pos in text_positions:
            x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
            if x < 0 or y < 0 or x + w > self.image_rgb.shape[1] or y + h > self.image_rgb.shape[0]: continue
            text_region = self.image_rgb[y:y+h, x:x+w]
            if text_region.size == 0: continue
            gray_region = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray_region)
            readability_scores.append(min(contrast / 128, 1.0))
        return np.mean(readability_scores) if readability_scores else 0.5

    def _analyze_text_hierarchy(self, font_sizes: List[int]) -> float:
        if len(font_sizes) < 2: return 1.0 if font_sizes else 0.5
        unique_sizes = len(set(font_sizes))
        if 2 <= unique_sizes <= 4: return 1.0
        return 0.3 if unique_sizes == 1 else 0.6

    def _analyze_text_positioning(self, text_positions: List[Dict]) -> float:
        if not text_positions: return 1.0
        h, w = self.image_gray.shape
        counts = [
            sum(1 for p in text_positions if p['y'] < h * 0.3),
            sum(1 for p in text_positions if h * 0.3 <= p['y'] < h * 0.7),
            sum(1 for p in text_positions if p['y'] >= h * 0.7)
        ]
        return 1.0 - max(counts) / len(text_positions)

    def _analyze_text_contrast(self, text_positions: List[Dict]) -> float:
        if not text_positions: return 1.0
        contrast_scores = []
        for pos in text_positions:
            x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
            if x < 0 or y < 0 or x + w > self.image_rgb.shape[1] or y + h > self.image_rgb.shape[0]: continue
            text_region = self.image_gray[y:y+h, x:x+w]
            if text_region.size == 0: continue
            bg_margin = 10
            bg_x1, bg_y1 = max(0, x - bg_margin), max(0, y - bg_margin)
            bg_x2, bg_y2 = min(self.image_gray.shape[1], x + w + bg_margin), min(self.image_gray.shape[0], y + h + bg_margin)
            bg_region = self.image_gray[bg_y1:bg_y2, bg_x1:bg_x2]
            contrast = abs(np.mean(text_region) - np.mean(bg_region)) / 255.0
            contrast_scores.append(contrast)
        return np.mean(contrast_scores) if contrast_scores else 0.5

    def _detect_cta_elements(self, texts: List[str]) -> bool:
        cta_keywords = ['купить', 'заказать', 'скачать', 'узнать', 'buy', 'order', 'download', 'get']
        all_text = ' '.join(texts).lower()
        return any(keyword in all_text for keyword in cta_keywords)

    def _calculate_aspect_ratio(self) -> float:
        if self.image is None: return 1.0
        w, h = self.image.size
        return w / h if h > 0 else 1.0

    def _calculate_size_score(self) -> float:
        if self.image is None: return 0.5
        w, h = self.image.size
        total_pixels = w * h
        return min(total_pixels / (1920 * 1080), 2.0) / 2.0
