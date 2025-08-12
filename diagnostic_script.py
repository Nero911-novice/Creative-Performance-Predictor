# check_dependencies.py
"""
Диагностический скрипт для проверки зависимостей Creative Performance Predictor.
Запустите перед первым запуском приложения.
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple

def check_python_version() -> bool:
    """Проверка версии Python."""
    version = sys.version_info
    print(f"🐍 Python версия: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Версия Python подходит")
        return True
    else:
        print("❌ Требуется Python 3.8 или выше")
        return False

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Проверка установки пакета."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True, "✅ Установлен"
    except ImportError as e:
        return False, f"❌ Не установлен: {e}"

def install_package(package_name: str) -> bool:
    """Установка пакета."""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name, "--quiet"
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Основная функция диагностики."""
    print("🔍 Creative Performance Predictor - Диагностика зависимостей")
    print("=" * 70)
    
    # Проверка Python
    if not check_python_version():
        print("\n🛑 Критическая ошибка: обновите Python до версии 3.8+")
        return
    
    print("\n📦 Проверка основных зависимостей:")
    
    # Основные пакеты
    core_packages = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("Pillow", "PIL"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("scipy", "scipy"),
        ("joblib", "joblib")
    ]
    
    missing_core = []
    
    for package, import_name in core_packages:
        success, message = check_package(import_name)
        print(f"  {package}: {message}")
        if not success:
            missing_core.append(package)
    
    print("\n📦 Проверка продвинутых зависимостей:")
    
    # Продвинутые пакеты
    advanced_packages = [
        ("opencv-python-headless", "cv2"),
        ("xgboost", "xgboost"),
        ("seaborn", "seaborn"),
        ("easyocr", "easyocr"),
        ("pytesseract", "pytesseract"),
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("torchvision", "torchvision")
    ]
    
    missing_advanced = []
    working_advanced = []
    
    for package, import_name in advanced_packages:
        success, message = check_package(import_name)
        print(f"  {package}: {message}")
        if not success:
            missing_advanced.append(package)
        else:
            working_advanced.append(package)
    
    print("\n📋 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ:")
    print("=" * 70)
    
    if not missing_core:
        print("✅ Все основные зависимости установлены!")
    else:
        print("❌ Отсутствуют основные зависимости:")
        for pkg in missing_core:
            print(f"   - {pkg}")
    
    if working_advanced:
        print(f"\n✅ Работающие продвинутые функции ({len(working_advanced)}):")
        for pkg in working_advanced:
            print(f"   - {pkg}")
    
    if missing_advanced:
        print(f"\n⚠️ Отсутствующие продвинутые функции ({len(missing_advanced)}):")
        for pkg in missing_advanced:
            print(f"   - {pkg}")
        print("   (Приложение будет работать с ограниченным функционалом)")
    
    # Проверка модулей проекта
    print("\n🔧 Проверка модулей проекта:")
    
    project_modules = [
        "config",
        "image_analyzer", 
        "ml_engine",
        "visualizer",
        "recommender"
    ]
    
    missing_modules = []
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}.py")
        except ImportError as e:
            print(f"  ❌ {module}.py: {e}")
            missing_modules.append(module)
    
    # Итоговые рекомендации
    print("\n🚀 РЕКОМЕНДАЦИИ:")
    print("=" * 70)
    
    if missing_core:
        print("1. Установите основные зависимости:")
        print("   pip install streamlit pandas numpy scikit-learn Pillow matplotlib plotly scipy joblib")
    
    if missing_advanced:
        print("2. Для полного функционала установите:")
        print("   pip install opencv-python-headless xgboost seaborn")
        print("   pip install easyocr pytesseract ultralytics  # Опционально")
    
    if missing_modules:
        print("3. Проверьте наличие файлов модулей в папке проекта:")
        for module in missing_modules:
            print(f"   - {module}.py")
    
    if not missing_core and not missing_modules:
        print("✅ Система готова к запуску!")
        print("   Запустите: streamlit run main.py")
    else:
        print("⚠️ Исправьте ошибки перед запуском")
    
    print("\n💡 Для автоматической установки используйте:")
    print("   python install_script.py")

if __name__ == "__main__":
    main()
