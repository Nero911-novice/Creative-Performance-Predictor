# install.py
"""
Скрипт автоматической установки зависимостей для Creative Performance Predictor.
Запустите: python install.py
"""

import subprocess
import sys
import importlib

def install_package(package):
    """Установка пакета через pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package_name, import_name=None):
    """Проверка установлен ли пакет."""
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def main():
    print("🎨 Creative Performance Predictor - Автоустановка зависимостей")
    print("=" * 60)
    
    # Обязательные пакеты
    required_packages = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("plotly", "plotly"),
        ("pillow", "PIL"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
        ("joblib", "joblib")
    ]
    
    # Опциональные пакеты
    optional_packages = [
        ("opencv-python-headless==4.5.5.64", "cv2"),
        ("pytesseract", "pytesseract")
    ]
    
    print("📦 Установка обязательных пакетов...")
    
    failed_required = []
    for package, import_name in required_packages:
        if not check_package(import_name):
            print(f"⏳ Устанавливаю {package}...")
            if install_package(package):
                print(f"✅ {package} установлен успешно")
            else:
                print(f"❌ Ошибка установки {package}")
                failed_required.append(package)
        else:
            print(f"✅ {package} уже установлен")
    
    print("\n📦 Установка опциональных пакетов...")
    
    failed_optional = []
    for package, import_name in optional_packages:
        if not check_package(import_name):
            print(f"⏳ Устанавливаю {package}...")
            if install_package(package):
                print(f"✅ {package} установлен успешно")
            else:
                print(f"⚠️ Не удалось установить {package} (не критично)")
                failed_optional.append(package)
        else:
            print(f"✅ {package} уже установлен")
    
    print("\n" + "=" * 60)
    print("📋 РЕЗУЛЬТАТЫ УСТАНОВКИ:")
    
    if not failed_required:
        print("✅ Все обязательные зависимости установлены успешно!")
    else:
        print("❌ Не удалось установить:")
        for package in failed_required:
            print(f"   - {package}")
    
    if failed_optional:
        print("⚠️ Опциональные пакеты не установлены:")
        for package in failed_optional:
            print(f"   - {package}")
        print("   (Приложение будет работать с ограниченным функционалом)")
    
    print("\n🚀 Для запуска приложения выполните:")
    print("   streamlit run main.py")
    
    if failed_required:
        print("\n🛠️ Для ручной установки проблемных пакетов:")
        for package in failed_required:
            print(f"   pip install {package}")

if __name__ == "__main__":
    main()