import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

data = pd.read_csv('car_price_prediction_.csv')
print("Размер датасета:", data.shape)
print("\nПервые 5 строк:")
print(data.head())


print("Пропуски в данных:")
print(data.isnull().sum())
print(f"\nДубликаты: {data.duplicated().sum()}")

print("\nОсновная информация о данных:")
print(data.info())
print("\nОписательная статистика:")
print(data.describe())

data['Car ID'] = data['Car ID'].astype(str)
data['Year'] = data['Year'].astype(int)

current_year = 2024
data['Car_Age'] = current_year - data['Year']
data['Price_per_Engine'] = data['Price'] / data['Engine Size']
data['Mileage_Category'] = pd.cut(data['Mileage'],
                                  bins=[0, 50000, 100000, 200000, 300000, np.inf],
                                  labels=['Очень низкий', 'Низкий', 'Средний', 'Высокий', 'Очень высокий'])

print("Новые признаки созданы")


plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(data['Price'], bins=50, alpha=0.7, color='skyblue')
plt.title('Распределение цен')
plt.xlabel('Цена')
plt.ylabel('Частота')

plt.subplot(2, 3, 2)
sns.boxplot(x='Fuel Type', y='Price', data=data)
plt.title('Цены по типам топлива')
plt.xticks(rotation=45)

plt.subplot(2, 3, 3)
sns.scatterplot(x='Mileage', y='Price', hue='Fuel Type', data=data, alpha=0.6)
plt.title('Зависимость цены от пробега')

plt.subplot(2, 3, 4)
brand_price = data.groupby('Brand')['Price'].mean().sort_values(ascending=False)
brand_price.plot(kind='bar', color='lightgreen')
plt.title('Средняя цена по брендам')
plt.xticks(rotation=45)

plt.subplot(2, 3, 5)
condition_counts = data['Condition'].value_counts()
plt.pie(condition_counts.values, labels=condition_counts.index, autopct='%1.1f%%')
plt.title('Распределение по состоянию')

plt.subplot(2, 3, 6)
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
brand_counts = data['Brand'].value_counts()
sns.barplot(x=brand_counts.values, y=brand_counts.index, palette='viridis')
plt.title('Количество автомобилей по брендам')
plt.xlabel('Количество')

plt.subplot(2, 2, 2)
fuel_transmission = pd.crosstab(data['Fuel Type'], data['Transmission'])
sns.heatmap(fuel_transmission, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Тип топлива vs Трансмиссия')

plt.subplot(2, 2, 3)
year_price_trend = data.groupby('Year')['Price'].mean()
plt.plot(year_price_trend.index, year_price_trend.values, marker='o', linewidth=2)
plt.title('Динамика средней цены по годам')
plt.xlabel('Год')
plt.ylabel('Средняя цена')

plt.subplot(2, 2, 4)
engine_price = data.groupby(pd.cut(data['Engine Size'], bins=10))['Price'].mean()
engine_price.plot(kind='bar', color='orange')
plt.title('Средняя цена по объему двигателя')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


def abc_xyz_analysis(data):
    data_sorted = data.sort_values('Price', ascending=False)
    data_sorted['Cumulative_Sum'] = data_sorted['Price'].cumsum()
    data_sorted['Cumulative_Percentage'] = (data_sorted['Cumulative_Sum'] / data_sorted['Price'].sum()) * 100

    def abc_category(x):
        if x <= 80:
            return 'A'
        elif x <= 95:
            return 'B'
        else:
            return 'C'

    data_sorted['ABC_Category'] = data_sorted['Cumulative_Percentage'].apply(abc_category)

    brand_stability = data.groupby('Brand')['Price'].std() / data.groupby('Brand')['Price'].mean()

    def xyz_category(x):
        if x <= 0.1:
            return 'X'
        elif x <= 0.25:
            return 'Y'
        else:
            return 'Z'

    brand_stability = brand_stability.apply(xyz_category)

    return data_sorted, brand_stability


abc_data, xyz_categories = abc_xyz_analysis(data)

print("ABC анализ (топ-20 автомобилей):")
print(abc_data[['Car ID', 'Brand', 'Model', 'Price', 'ABC_Category']].head(20))

print("\nXYZ анализ по брендам:")
print(xyz_categories)


def automotive_rfm_analysis(data):
    data['Recency'] = current_year - data['Year']

    brand_frequency = data['Brand'].value_counts()
    data['Frequency_Score'] = data['Brand'].map(brand_frequency)

    data['Monetary_Score'] = data['Price']

    scaler = StandardScaler()
    rfm_scores = data[['Recency', 'Frequency_Score', 'Monetary_Score']].copy()
    rfm_scores[['Recency', 'Frequency_Score', 'Monetary_Score']] = scaler.fit_transform(
        rfm_scores[['Recency', 'Frequency_Score', 'Monetary_Score']])

    kmeans = KMeans(n_clusters=4, random_state=42)
    data['RFM_Cluster'] = kmeans.fit_predict(rfm_scores)

    cluster_profile = data.groupby('RFM_Cluster').agg({
        'Recency': 'mean',
        'Frequency_Score': 'mean',
        'Monetary_Score': 'mean',
        'Brand': 'count'
    }).round(2)

    return data, cluster_profile


rfm_data, cluster_profile = automotive_rfm_analysis(data)

print("RFM сегментация - профили кластеров:")
print(cluster_profile)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Monetary_Score', y='Recency', hue='RFM_Cluster', data=rfm_data, palette='viridis')
plt.title('RFM Анализ: Ценность vs Новизна')

plt.subplot(1, 2, 2)
cluster_counts = rfm_data['RFM_Cluster'].value_counts().sort_index()
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='coolwarm')
plt.title('Распределение по RFM сегментам')
plt.xlabel('RFM Сегмент')
plt.ylabel('Количество автомобилей')

plt.tight_layout()
plt.show()

premium_threshold = data['Price'].quantile(0.8)
premium_cars = data[data['Price'] >= premium_threshold]

print("АНАЛИЗ ПРЕМИУМ СЕГМЕНТА:")
print(f"Количество премиум автомобилей: {len(premium_cars)}")
print(f"Доля премиум сегмента: {len(premium_cars) / len(data) * 100:.1f}%")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
premium_brands = premium_cars['Brand'].value_counts()
sns.barplot(x=premium_brands.values, y=premium_brands.index, palette='RdYlBu_r')
plt.title('Бренды в премиум сегменте')

plt.subplot(1, 3, 2)
premium_fuel = premium_cars['Fuel Type'].value_counts()
plt.pie(premium_fuel.values, labels=premium_fuel.index, autopct='%1.1f%%')
plt.title('Типы топлива в премиум сегменте')

plt.subplot(1, 3, 3)
sns.boxplot(x='Transmission', y='Price', data=premium_cars)
plt.title('Цены премиум авто по типу трансмиссии')

plt.tight_layout()
plt.show()

budget_threshold = data['Price'].quantile(0.2)
budget_cars = data[data['Price'] <= budget_threshold]

print("\nАНАЛИЗ ЭКОНОМИЧНОГО СЕГМЕНТА:")
print(f"Количество бюджетных автомобилей: {len(budget_cars)}")

summary_table = data.groupby('Brand').agg({
    'Price': ['mean', 'median', 'count'],
    'Mileage': 'mean',
    'Year': 'mean',
    'Engine Size': 'mean'
}).round(2)

print("\nСВОДНАЯ ТАБЛИЦА ПО БРЕНДАМ:")
print(summary_table.head(10))

print("=" * 80)
print("КЛЮЧЕВЫЕ ВЫВОДЫ И ИНСАЙТЫ")
print("=" * 80)

insights = [
    f"1. ОБЩИЙ ОБЗОР: Всего {len(data)} автомобилей, {data['Brand'].nunique()} брендов",
    f"2. ЦЕНОВОЙ ДИАПАЗОН: от ${data['Price'].min():,.0f} до ${data['Price'].max():,.0f}",
    f"3. САМЫЕ ПРЕДСТАВЛЕННЫЕ БРЕНДЫ: {', '.join(data['Brand'].value_counts().head(3).index.tolist())}",
    f"4. СРЕДНИЙ ПРОБЕГ: {data['Mileage'].mean():,.0f} км",
    f"5. РАСПРЕДЕЛЕНИЕ ПО ТОПЛИВУ: {data['Fuel Type'].value_counts().to_dict()}",
    f"6. ПРЕМИУМ СЕГМЕНТ: {len(premium_cars)} авто ({len(premium_cars) / len(data) * 100:.1f}% от общего числа)",
    f"7. ABC АНАЛИЗ: Категория A включает {len(abc_data[abc_data['ABC_Category'] == 'A'])} наиболее ценных авто",
    f"8. RFM СЕГМЕНТАЦИЯ: Выявлено {rfm_data['RFM_Cluster'].nunique()} различных сегментов покупателей"
]

for insight in insights:
    print(f"• {insight}")

print("\nРЕКОМЕНДАЦИИ:")
recommendations = [
    f"1. Сфокусироваться на премиальных брендах {', '.join(premium_brands.index[:3])} для максимизации прибыли",
    "2. Развивать сегмент электромобилей как перспективное направление",
    "3. Оптимизировать ценовую политику для разных RFM сегментов",
    "4. Увеличить представленность популярных моделей в экономичном сегменте",
    "5. Внедрить дифференцированный маркетинг для разных категорий ABC анализа"
]

for rec in recommendations:
    print(f"✓ {rec}")
