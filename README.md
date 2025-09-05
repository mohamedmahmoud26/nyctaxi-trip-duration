# NYC Taxi Trip Duration Prediction | توقع مدة رحلات التاكسي في نيويورك

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Active-brightgreen)

**English:**
This project predicts the duration of NYC taxi trips using machine learning. It includes data processing, feature engineering, model training, and evaluation, with all code and results organized for easy use and reproducibility.

**بالعربي:**
المشروع ده بيستخدم الذكاء الاصطناعي لتوقع مدة رحلات التاكسي في نيويورك بناءً على بيانات فعلية، مع معالجة وتحليل البيانات وتدريب النماذج وعرض النتائج بشكل منظم.

---

## Project Structure | هيكل المشروع

```
asset/                # رسوم ونتائج التحليل
  └── graphs/         # صور ورسوم بيانية
data/                 # البيانات الخام
  └── data_split0/    # تقسيمات البيانات (train, val, test)
data_processed/       # بيانات معالجة
models/               # النماذج المدربة
Notebook/             # دفاتر Jupyter للتحليل
results/              # نتائج النمذجة والتقييم
src/                  # كود المشروع (معالجة، تدريب، تقييم...)
submissions/          # ملفات التقديم النهائية
requirement.txt       # المتطلبات البرمجية
LICENSE               # الرخصة
README.md             # ملف الشرح
```

---

## Quick Start | خطوات التشغيل السريعة

1. **تفعيل البيئة الافتراضية (conda):**
	```bash
	conda activate env1
	```
2. **تثبيت المتطلبات:**
	```bash
	pip install -r requirement.txt
	```
3. **تشغيل التدريب أو التقييم:**
	```bash
	python src/main.py
	```
4. **استكشاف البيانات:**
	- افتح `Notebook/EDA_TripDuration.ipynb` باستخدام Jupyter Notebook.

---

## النتائج (Results)

- أهم الميزات المؤثرة: `results/top10_shap_features.csv`
- ترتيب أهمية الميزات: `results/shap_feature_importance.csv`
- نتائج التحقق المتقاطع: `results/cv_scores.csv`
- توقعات التحقق: `results/validation_predictions.csv`
- أفضل نموذج: Ridge Regression (`models/ridge_pipeline.pkl`)

---

## المساهمة (Contributing)

1. Fork المشروع
2. أنشئ فرع جديد للتعديل
	```bash
	git checkout -b feature/اسم_الميزة
	```
3. نفذ تعديلاتك ثم commit
	```bash
	git commit -m "شرح التعديل"
	```
4. ادفع الفرع (push) وافتح Pull Request

---

## الشكر (Acknowledgments)

- NYC Taxi and Limousine Commission على البيانات
- مجتمع المصادر المفتوحة على الأدوات والمكتباتAPI branch test
API branch test
