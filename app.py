"""
Spotify 音乐特征交互式可视化分析平台 (适度降级版)
保留基础Tab、基础布局、基础图表；删除平行坐标、3D散点、气泡、调性热力、手肘图、歌曲推荐等高加分模块
页面恢复白色默认主题，移除多余文字提示
运行方式：streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ======================== 页面基础配置（默认白色主题） ========================
st.set_page_config(
    page_title="Spotify 音乐特征深度分析",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== 数据加载缓存 ========================
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_tracks.csv")
    df['Release Date'] = pd.to_datetime(df['Release Date'])
    df['Year'] = df['Release Date'].dt.year
    df['Duration_min'] = df['Duration_ms'] / 60000
    df['Popularity_Level'] = pd.cut(
        df['Popularity'],
        bins=[-1, 25, 50, 75, 101],
        labels=['小众(0-25)', '一般(25-50)', '热门(50-75)', '爆款(75-100)']
    )
    df['Tempo_Level'] = pd.cut(
        df['Tempo'],
        bins=[0, 80, 110, 140, 250],
        labels=['慢速', '中速', '快速', '极快']
    )
    return df

df = load_data()

# ======================== 基础常量 ========================
FEATURES = ['Danceability', 'Energy', 'Loudness', 'Speechiness',
            'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
FEATURE_LABELS = {
    'Danceability': '舞动性', 'Energy': '能量', 'Loudness': '响度',
    'Speechiness': '人声度', 'Acousticness': '原声度',
    'Instrumentalness': '器乐度', 'Liveness': '现场感',
    'Valence': '效价(积极性)', 'Tempo': 'BPM'
}
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ======================== 侧边筛选栏（完整保留） ========================
with st.sidebar:
    st.markdown("筛选控制台")
    all_genres = sorted(df['Genre'].unique())
    selected_genres = st.multiselect("流派选择", options=all_genres, default=list(all_genres))
    pop_min, pop_max = st.slider("流行度区间", 0, 100, (0, 100), 5)
    year_min_val, year_max_val = st.slider("发行年份", int(df['Year'].min()), int(df['Year'].max()),
                                           (int(df['Year'].min()), int(df['Year'].max())))
    explicit_filter = st.radio("Explicit", ['全部', '仅 Explicit', '仅非 Explicit'], horizontal=True)

# 筛选逻辑
filtered_df = df[df['Genre'].isin(selected_genres)] if selected_genres else df
filtered_df = filtered_df[(filtered_df['Popularity'] >= pop_min) & (filtered_df['Popularity'] <= pop_max)]
filtered_df = filtered_df[(filtered_df['Year'] >= year_min_val) & (filtered_df['Year'] <= year_max_val)]
if explicit_filter == '仅 Explicit':
    filtered_df = filtered_df[filtered_df['Explicit'] == True]
elif explicit_filter == '仅非 Explicit':
    filtered_df = filtered_df[filtered_df['Explicit'] == False]

# ======================== 标题 ========================
st.title("Spotify 音乐特征深度分析平台")
st.markdown("---")

# ======================== 五Tab完整保留 ========================
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "数据概览", "特征分析", "关联探索", "聚类与推荐", "数据明细"
])

# ======================== TAB0 数据概览 ========================
with tab0:
    st.subheader("📊 关键指标")
    cols = st.columns(8)
    metrics = [
        ("总曲目", f"{len(filtered_df):,}"),
        ("均流行度", f"{filtered_df['Popularity'].mean():.1f}"),
        ("均舞动性", f"{filtered_df['Danceability'].mean():.3f}"),
        ("均能量", f"{filtered_df['Energy'].mean():.3f}"),
        ("均效价", f"{filtered_df['Valence'].mean():.3f}"),
        ("均BPM", f"{filtered_df['Tempo'].mean():.0f}"),
        ("艺术家", f"{filtered_df['Artist'].nunique()}"),
        ("Explicit%", f"{filtered_df['Explicit'].mean()*100:.1f}"),
    ]
    for col, (label, val) in zip(cols, metrics):
        with col:
            st.metric(label, val)
    st.markdown("---")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("**流派层级分布 — 旭日图**")
        genre_pop = filtered_df.groupby('Genre').agg(count=('Track ID', 'count'), avg_pop=('Popularity', 'mean')).reset_index()
        genre_pop['level'] = 'All Genres'
        fig_sun = px.sunburst(genre_pop, path=['level', 'Genre'], values='count', color='avg_pop', labels={'count': '曲目数', 'avg_pop': '平均流行度'})
        st.plotly_chart(fig_sun, use_container_width=True)
    with col_b:
        st.markdown("**各流派流行度 — 箱线图**")
        fig_box = px.box(filtered_df, x='Genre', y='Popularity', labels={'Popularity': '流行度', 'Genre': '流派'}, points='outliers')
        st.plotly_chart(fig_box, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**平均BPM — 仪表盘**")
        avg_tempo = filtered_df['Tempo'].mean()
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number+delta", value=avg_tempo, number=dict(suffix=" BPM"), title=dict(text="当前筛选下平均BPM"), gauge=dict(axis=dict(range=[50, 200]))))
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col_d:
        st.markdown("**流行度分级分布 — 漏斗图**")
        pop_dist = filtered_df['Popularity_Level'].value_counts().reindex(['爆款(75-100)', '热门(50-75)', '一般(25-50)', '小众(0-25)'])
        fig_funnel = go.Figure(go.Funnel(y=pop_dist.index.tolist(), x=pop_dist.values.tolist()))
        st.plotly_chart(fig_funnel, use_container_width=True)

# ======================== TAB1 特征分析（空白列无文字） ========================
with tab1:
    st.subheader("音频特征多维分析")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**各流派音频特征分布 — 小提琴图**")
        violin_feat = st.selectbox("选择特征", FEATURES, index=0, format_func=lambda f: FEATURE_LABELS.get(f, f))
        fig_violin = px.violin(filtered_df, x='Genre', y=violin_feat, box=True, points='outliers')
        st.plotly_chart(fig_violin, use_container_width=True)
    # 右侧空白，无任何文字
    with col_b:
        pass

    col_c, col_d = st.columns([3, 2])
    with col_c:
        st.markdown("**特征相关性矩阵 — 热力图**")
        corr_features = FEATURES + ['Popularity', 'Duration_min']
        corr = filtered_df[corr_features].corr()
        fig_heat = px.imshow(corr, text_auto='.2f', zmin=-1, zmax=1)
        st.plotly_chart(fig_heat, use_container_width=True)
    with col_d:
        st.markdown("**流行度相关系数排名**")
        pop_corrs = []
        for feat in FEATURES:
            r = filtered_df[feat].corr(filtered_df['Popularity'])
            pop_corrs.append((FEATURE_LABELS.get(feat), r))
        pop_corrs.sort(key=lambda x: x[1])
        fig_bar = go.Figure(go.Bar(x=[r for _, r in pop_corrs], y=[n for n, _ in pop_corrs], orientation='h'))
        st.plotly_chart(fig_bar, use_container_width=True)

# ======================== TAB2 关联探索（两列空白，不输出文字） ========================
with tab2:
    st.subheader("多维关联探索")
    col_a, col_b = st.columns(2)
    with col_a:
        pass
    with col_b:
        pass
    st.markdown("---")
    col_c, col_d = st.columns(2)
    with col_c:
        pass
    with col_d:
        st.markdown("**歌曲时长与流行度关系**")
        dur_bins = [0, 2, 2.5, 3, 3.5, 4, 5, 15]
        dur_labels = ['<2分', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-5', '>5分']
        filtered_df_cp = filtered_df.copy()
        filtered_df_cp['Dur_Bin'] = pd.cut(filtered_df_cp['Duration_min'], bins=dur_bins, labels=dur_labels)
        dur_stats = filtered_df_cp.groupby('Dur_Bin', observed=False).agg(Count=('Track ID', 'count'), Avg_Pop=('Popularity', 'mean')).reset_index()
        fig_dur = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dur.add_trace(go.Bar(x=dur_stats['Dur_Bin'], y=dur_stats['Count']), secondary_y=False)
        fig_dur.add_trace(go.Scatter(x=dur_stats['Dur_Bin'], y=dur_stats['Avg_Pop']), secondary_y=True)
        st.plotly_chart(fig_dur, use_container_width=True)

# ======================== TAB3 聚类分析（删掉多余文字提示） ========================
with tab3:
    st.subheader("K-Means 聚类分析")
    cluster_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Instrumentalness', 'Tempo']
    X = filtered_df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=2024, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    filtered_df['cluster'] = labels
    st.write("三类歌曲数量统计：")
    st.write(filtered_df['cluster'].value_counts())

# ======================== TAB4 数据明细表 ========================
with tab4:
    st.subheader("数据明细表")
    search_data = st.text_input("搜索曲目或艺术家")
    sort_col = st.selectbox("排序字段", ['Popularity', 'Danceability', 'Energy', 'Valence', 'Tempo', 'Duration_min'])
    sort_dir = st.radio("排序方向", ['降序', '升序'], horizontal=True)
    display_cols = ['Track ID', 'Track Name', 'Artist', 'Genre', 'Popularity', 'Danceability', 'Energy']
    display_df = filtered_df[display_cols].copy()
    if search_data:
        display_df = display_df[
            display_df['Track Name'].str.contains(search_data, case=False, na=False) |
            display_df['Artist'].str.contains(search_data, case=False, na=False)
        ]
    display_df = display_df.sort_values(sort_col, ascending=(sort_dir == '升序'))
    st.dataframe(display_df, use_container_width=True)
    st.caption(f"共 {len(display_df)} 条记录")

# 页脚
st.markdown("---")
st.caption("数据可视化课程期末作业")