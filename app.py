import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="书店经营决策助手", layout="wide")
st.title("📚 书店经营决策助手")

# ---------- 加载数据 ----------
@st.cache_data
def load_books():
    if os.path.exists('enhanced_books.csv'):
        return pd.read_csv('enhanced_books.csv')
    st.error("未找到 enhanced_books.csv")
    st.stop()
books_df = load_books()

# ---------- 加载默认销量预测模型 ----------
@st.cache_resource
def load_default_model():
    if os.path.exists('best_model.pkl'):
        return joblib.load('best_model.pkl')
    return None
default_model = load_default_model()
if 'best_model' not in st.session_state:
    st.session_state['best_model'] = default_model

# ---------- 加载连带消费模型 ----------
@st.cache_resource
def load_attachment_models():
    try:
        coffee_model = joblib.load('coffee_model_optimized.pkl')
        cultural_model = joblib.load('cultural_model_optimized.pkl')
        encoders = joblib.load('encoders_optimized.pkl')
        return coffee_model, cultural_model, encoders
    except FileNotFoundError:
        st.warning("连带消费模型文件未找到，将使用固定比率（咖啡35%，文创12%）")
        return None, None, None

coffee_model, cultural_model, encoders = load_attachment_models()

# ---------- 加载协同过滤模型 ----------
@st.cache_resource
def load_cf():
    if os.path.exists('item_sim_real.pkl') and os.path.exists('books_for_rec_real.csv'):
        item_sim = joblib.load('item_sim_real.pkl')
        books_rec = pd.read_csv('books_for_rec_real.csv')
        return item_sim, books_rec, "真实 Book-Crossing"
    elif os.path.exists('item_sim.pkl') and os.path.exists('books_for_rec.csv'):
        item_sim = joblib.load('item_sim.pkl')
        books_rec = pd.read_csv('books_for_rec.csv')
        return item_sim, books_rec, "基于属性"
    else:
        return None, None, None
item_sim, books_rec, cf_source = load_cf()

# ---------- 辅助函数 ----------
def clean_title(title):
    """清理书名用于匹配：去除括号内容、多余空格，取前20字符小写"""
    if not isinstance(title, str):
        return ""
    title = re.sub(r'[（(].*?[）)]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title[:20].lower()

def reorder_suggestion(book, predicted_sales):
    daily = max(predicted_sales / 30, 0.1)
    stock = book.get('stock', 50)
    if stock < daily * 14:
        need = int(daily * 21 - stock)
        return f"⚠️ 库存不足，建议补货 {need} 本"
    else:
        days = int(stock / daily)
        return f"✅ 库存充足，可维持 {days} 天"

def predict_sales(book, user_price, marketing_budget):
    online_price = book.get('onlinePrice', book['listPrice'] * 0.6)
    price_ratio = user_price / online_price if online_price > 0 else 1.0
    model = st.session_state.get('best_model', default_model)
    if model is None:
        base_sales = 50
        sales = base_sales * (1 - max(0, price_ratio - 0.9) * 1.2)
        sales = max(5, int(sales))
        sales += int(marketing_budget) * 10
        return sales
    try:
        X = [[price_ratio, 0, marketing_budget, book['rating'], book.get('reviews', 0), book['listPrice']]]
        pred = model.predict(X)[0]
        return max(1, int(pred))
    except Exception:
        base_sales = 50
        sales = base_sales * (1 - max(0, price_ratio - 0.9) * 1.2)
        sales = max(5, int(sales))
        sales += int(marketing_budget) * 10
        return sales

def predict_attachment_optimized(book, n_customers, is_weekend=0, hour_of_day=14):
    if coffee_model is None or cultural_model is None or encoders is None:
        coffee_sales = int(n_customers * 0.35)
        cultural_sales = int(n_customers * 0.12)
        return coffee_sales, cultural_sales
    try:
        import numpy as np
        import pandas as pd

        cust_types = ['新客', '老客', '会员']
        cust_weights = [0.30, 0.50, 0.20]
        dwell_times = np.random.exponential(30, size=n_customers)

        features = []
        for ct, dwell in zip(np.random.choice(cust_types, n_customers, p=cust_weights), dwell_times):
            cat_enc = encoders['book_category'].transform([book.get('category', '其他')])[0]
            cust_enc = encoders['customer_type'].transform([ct])[0]
            features.append([
                book['listPrice'],
                book['rating'],
                is_weekend,
                hour_of_day,
                dwell,
                cat_enc,
                cust_enc
            ])

        X_pred = pd.DataFrame(features, columns=[
            'book_price', 'book_rating', 'is_weekend', 'hour_of_day',
            'dwell_time', 'book_category_enc', 'customer_type_enc'
        ])

        coffee_probs = coffee_model.predict_proba(X_pred)[:, 1]
        cultural_probs = cultural_model.predict_proba(X_pred)[:, 1]

        coffee_sales = int(np.sum(np.random.binomial(1, coffee_probs)))
        cultural_sales = int(np.sum(np.random.binomial(1, cultural_probs)))
        return coffee_sales, cultural_sales
    except Exception:
        return int(n_customers * 0.35), int(n_customers * 0.12)

# ---------- 重训练函数（使用模糊匹配） ----------
def retrain_model(df):
    try:
        if len(df) < 5:
            st.error(f"数据量太少（{len(df)} 条），至少需要 5 条记录")
            return None, 0, 0

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['is_weekend'] = df['date'].dt.dayofweek >= 5
        else:
            df['is_weekend'] = 0

        # 创建匹配键
        df['match_key'] = df['title'].apply(clean_title)
        books_df['match_key'] = books_df['title'].apply(clean_title)

        print("上传数据前3个匹配键:", df['match_key'].head(3).tolist())
        print("系统图书前3个匹配键:", books_df['match_key'].head(3).tolist())

        merged = df.merge(books_df, on='match_key', how='left')
        merged.drop('match_key', axis=1, inplace=True)
        merged = merged.dropna(subset=['rating', 'listPrice'])

        if len(merged) < 5:
            st.error(f"匹配后有效数据不足（{len(merged)} 条），请检查书名是否匹配")
            unmatched = df[~df['match_key'].isin(books_df['match_key'])]['title'].unique()
            if len(unmatched) > 0:
                st.write("以下书名未能匹配：", list(unmatched)[:5])
            return None, 0, 0

        merged['price_ratio'] = merged['price'] / (merged['listPrice'] * 0.6)
        features = ['price_ratio', 'is_weekend', 'marketing', 'rating', 'reviews', 'listPrice']
        X = merged[features]
        y = merged['quantity']

        if len(merged) < 30:
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            model.fit(X, y)
            cv = min(3, len(merged))
            r2 = cross_val_score(model, X, y, cv=cv, scoring='r2').mean()
            mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error').mean()
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

        joblib.dump(model, 'best_model.pkl')
        return model, r2, mae
    except Exception as e:
        st.error(f"训练失败: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, 0, 0

# ---------- 侧边栏 ----------
st.sidebar.header("经营参数")
user_price = st.sidebar.slider("售价 (元)", 30, 100, 50, 5)
marketing_budget = st.sidebar.slider("营销预算 (千元)", 0.5, 3.0, 1.5, 0.1)
rent = st.sidebar.slider("租金 (元)", 800, 5000, 1000, 100)
staff_count = st.sidebar.slider("店员人数", 0, 3, 0, 1)
avg_salary = st.sidebar.slider("店员月薪 (元)", 3000, 5000, 4000, 500)
utility = st.sidebar.slider("水电杂费 (元)", 300, 2000, 500, 100)
customer_coef = st.sidebar.slider("客流系数", 0.5, 2.0, 1.2, 0.1)

with st.sidebar.expander("📂 上传真实销售数据（CSV）"):
    uploaded = st.file_uploader("选择 CSV", type="csv")
    if uploaded:
        df_user = pd.read_csv(uploaded)
        st.write("数据预览", df_user.head())
        if st.button("重新训练模型"):
            with st.spinner("训练中，请稍候..."):
                new_model, r2, mae = retrain_model(df_user)
                if new_model:
                    st.session_state['best_model'] = new_model
                    st.success(f"训练完成！R²={r2:.4f}, MAE={mae:.2f}本")
                else:
                    st.error("训练失败，请检查数据格式")

    # 使用真实书名的示例数据
    if st.button("📂 加载示例数据并训练"):
        sample_titles = [
            "你当像鸟飞往你的山",
            "人间失格",
            "乌合之众 : 大众心理研究",
            "神奇校车・图画书版（全12册）",
            "作家榜名著：月亮与六便士",
            "人生海海",
            "正面管教(修订版)",
            "云边有个小卖部",
            "小熊和最好的爸爸（全7册）",
            "啊2.0"
        ]
        sample_data = []
        for i, title in enumerate(sample_titles, 1):
            sample_data.append({
                'date': f'2026-01-0{(i%3)+1}',
                'title': title,
                'quantity': 5 + i,
                'price': 45 + i,
                'marketing': 1.0 + i * 0.1
            })
        sample_df = pd.DataFrame(sample_data)
        with st.spinner("训练示例数据..."):
            new_model, r2, mae = retrain_model(sample_df)
            if new_model:
                st.session_state['best_model'] = new_model
                st.success(f"✅ 示例数据训练完成！R²={r2:.4f}, MAE={mae:.2f}本")
            else:
                st.error("示例数据训练失败，请检查控制台输出")

# ---------- 图书选择 ----------
st.subheader("📖 选择图书（可多选）")
search = st.text_input("🔍 搜索书名")
category_filter = st.selectbox("品类筛选", ['全部'] + list(books_df['category'].unique()))
filtered = books_df
if category_filter != '全部':
    filtered = filtered[filtered['category'] == category_filter]
if search:
    filtered = filtered[filtered['title'].str.contains(search, case=False)]

selected_titles = []
for i, (_, row) in enumerate(filtered.head(150).iterrows()):
    label = f"{row['title'][:50]} (¥{row['listPrice']})"
    if st.checkbox(label, key=f"book_{i}_{row['title'][:30]}"):
        selected_titles.append(row['title'])

# ---------- 生成报告 ----------
if st.button("🚀 生成经营决策报告", type="primary"):
    if not selected_titles:
        st.warning("请至少选择一本书")
    else:
        total_sales = total_rev = total_coffee = total_cultural = 0
        details = []
        first_book = None
        for title in selected_titles:
            book = books_df[books_df['title'] == title].iloc[0].to_dict()
            if first_book is None:
                first_book = book
            sales = predict_sales(book, user_price, marketing_budget)
            customers = int(sales * customer_coef)
            coffee, cultural = predict_attachment_optimized(book, customers)
            rev = sales * user_price
            total_sales += sales
            total_rev += rev
            total_coffee += coffee
            total_cultural += cultural
            reorder = reorder_suggestion(book, sales)
            details.append(f"- {book['title'][:40]} 销量{sales}本  咖啡{coffee}杯  文创{cultural}件  {reorder}")
        cafe_rev = total_coffee * 25
        cult_rev = total_cultural * 35
        total_income = total_rev + cafe_rev + cult_rev
        cost_total = total_rev * 0.6 + cafe_rev * 0.35 + cult_rev * 0.5
        fixed_cost = rent + utility + staff_count * avg_salary + marketing_budget * 1000
        net = total_income - cost_total - fixed_cost
        margin = (total_income - cost_total) / total_income if total_income > 0 else 0

        price_ratio = user_price / first_book.get('listPrice', 1)
        if price_ratio > 1.5:
            price_advice = "⚠️ 售价远超定价，强烈建议降价10-20%"
        elif price_ratio > 1.2:
            price_advice = "📉 售价偏高，建议降价5-10%"
        else:
            price_advice = "✅ 售价合理"

        st.subheader("📋 经营决策报告")
        for d in details:
            st.write(d)

        st.markdown("### 收入明细")
        col1, col2, col3 = st.columns(3)
        col1.metric("图书收入", f"¥{total_rev:.0f} ({total_sales}本 × {user_price}元)")
        col2.metric("咖啡收入", f"¥{cafe_rev:.0f} ({total_coffee}杯 × 25元)")
        col3.metric("文创收入", f"¥{cult_rev:.0f} ({total_cultural}件 × 35元)")

        st.markdown("### 财务健康")
        col1, col2 = st.columns(2)
        col1.metric("月总营收", f"¥{total_income:,.0f}")
        col1.metric("月净利润", f"¥{net:,.0f}", "盈利" if net >= 0 else "亏损")
        col2.metric("综合毛利率", f"{margin*100:.1f}%")
        col2.metric("日保本额", f"¥{fixed_cost / 30 / (margin if margin>0 else 0.01):.0f}")

        st.markdown("### 固定成本明细")
        st.write(f"租金: ¥{rent}   |   人工: {staff_count}人 × ¥{avg_salary} = ¥{staff_count*avg_salary}")
        st.write(f"水电: ¥{utility}   |   营销: ¥{marketing_budget*1000:.0f}")
        st.write(f"**合计固定成本: ¥{fixed_cost:,.0f}**")

        st.markdown("### 💡 相似推荐")
        if item_sim is not None and books_rec is not None:
            try:
                matched = books_rec[books_rec['title'].str.lower() == first_book['title'].lower()]
                if len(matched) > 0:
                    idx = matched.index[0]
                    sims = list(enumerate(item_sim[idx]))
                    sims.sort(key=lambda x: x[1], reverse=True)
                    rec_indices = [i[0] for i in sims[1:4]]
                    rec_books = books_rec.iloc[rec_indices]
                    st.write("**基于真实用户行为的推荐：**")
                    for _, r in rec_books.iterrows():
                        title = r['title']
                        match_row = books_df[books_df['title'].str.contains(title[:20], case=False, na=False)]
                        if len(match_row) > 0:
                            price = match_row.iloc[0]['listPrice']
                            rating = match_row.iloc[0]['rating']
                        else:
                            price = r.get('listPrice', 0)
                            rating = r.get('rating', 4.0)
                        st.write(f"• {title} (评分 {rating:.1f}, ¥{price:.0f})")
                else:
                    st.write("**基于同品类高评分的推荐：**")
                    same_cat = books_df[books_df['category'] == first_book['category']]
                    same_cat = same_cat[same_cat['title'] != first_book['title']]
                    if len(same_cat) > 0:
                        top_cat = same_cat.sort_values('rating', ascending=False).head(3)
                        for _, r in top_cat.iterrows():
                            st.write(f"• {r['title']} (评分 {r['rating']:.1f}, ¥{r['listPrice']})")
                    else:
                        st.write("暂无相似推荐")
            except Exception as e:
                st.write(f"推荐出错: {e}")
        else:
            st.write("协同过滤模型未加载，已启用基于品类的推荐")
            same_cat = books_df[books_df['category'] == first_book['category']]
            same_cat = same_cat[same_cat['title'] != first_book['title']]
            if len(same_cat) > 0:
                top_cat = same_cat.sort_values('rating', ascending=False).head(3)
                for _, r in top_cat.iterrows():
                    st.write(f"• {r['title']} (评分 {r['rating']:.1f}, ¥{r['listPrice']})")
            else:
                st.write("暂无相似推荐")

        st.markdown("### 🎯 模型性能指标")
        st.write("销量预测随机森林 R²=0.794, MAE=4.4本（通用模型）")
        if st.session_state.get('best_model') != default_model:
            st.success("✓ 当前销量预测模型已基于您上传的真实数据重新训练")

        st.markdown("### 📋 行动建议")
        st.write(f"1. {price_advice}")
        st.write(f"2. {'可尝试降低固定成本（租金/人力）' if net < 0 else '经营健康，可拓展品类'}")
        st.write("3. 重点关注高评分图书的陈列与营销")

        st.caption(f"报告时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
