
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Generate demo data
np.random.seed(42)
n_samples = 200

df = pd.DataFrame({
    'stock_qty': np.random.randint(10, 100, size=n_samples),
    'avg_daily_sales': np.random.uniform(0.5, 10.0, size=n_samples),
    'days_until_expiry': np.random.randint(5, 60, size=n_samples),
    'discount_applied': np.random.choice([0, 1], size=n_samples),
    'category': np.random.choice(['drink', 'snack', 'frozen', 'daily'], size=n_samples)
})

df = pd.get_dummies(df, columns=['category'], drop_first=True)
df['risky'] = (df['stock_qty'] / df['avg_daily_sales']) > df['days_until_expiry']
df['risky'] = df['risky'].astype(int)

X = df.drop(columns='risky')
y = df['risky']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# UI
st.title("🔍 ทำนายความเสี่ยงสินค้าขายไม่ทัน (สินค้าป้ายทอง)")
st.markdown("ระบบนี้ช่วยประเมินว่าสินค้ามีแนวโน้มจะขายไม่ทันหรือไม่ เพื่อช่วยติดป้ายทองเร่งขาย")

stock_qty = st.slider("จำนวนสินค้าคงเหลือ (ชิ้น)", 0, 200, 50)
avg_daily_sales = st.slider("ยอดขายเฉลี่ยต่อวัน", 0.1, 20.0, 2.0)
days_until_expiry = st.slider("จำนวนวันก่อนหมดอายุ", 1, 90, 14)
discount_applied = st.selectbox("มีการลดราคาหรือไม่", [0, 1], format_func=lambda x: "มี" if x == 1 else "ไม่มี")
category = st.selectbox("หมวดหมู่สินค้า", ["drink", "snack", "frozen", "daily"])

input_df = pd.DataFrame([{
    "stock_qty": stock_qty,
    "avg_daily_sales": avg_daily_sales,
    "days_until_expiry": days_until_expiry,
    "discount_applied": discount_applied,
    "category": category
}])

input_df = pd.get_dummies(input_df, columns=['category'])
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]

if st.button("ทำนายความเสี่ยง"):
    risk_prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error(f"⚠️ สินค้านี้เสี่ยงขายไม่ทัน! โอกาสเสี่ยง: {risk_prob:.0%}")
        st.markdown("👉 คำแนะนำ: ติด **ป้ายทอง** และใช้โปรโมชันกระตุ้นการขาย")
    else:
        st.success(f"✅ สินค้านี้ไม่น่าจะมีปัญหา ขายได้ทันเวลา (โอกาสเสี่ยง: {risk_prob:.0%})")
