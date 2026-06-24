"""
Spotify 音乐特征交互式可视化分析平台 (重构版)
基于 Streamlit + Plotly
深色主题 · Tab式布局 · 全新图表类型

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
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy import stats

# ======================== 页面配置 ========================
st.set_page_config(
    page_title="Spotify 音乐特征深度分析",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== 数据加载 ========================
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

# ======================== 常量定义 ========================
FEATURES = ['Danceability', 'Energy', 'Loudness', 'Speechiness',
            'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
FEATURE_LABELS = {
    'Danceability': '舞动性', 'Energy': '能量', 'Loudness': '响度',
    'Speechiness': '人声度', 'Acousticness': '原声度',
    'Instrumentalness': '器乐度', 'Liveness': '现场感',
    'Valence': '效价(积极性)', 'Tempo': 'BPM'
}
GENRE_COLORS = {
    'Pop': '#1db954', 'Rock': '#ff4757', 'Hip-Hop': '#ffa502',
    'Electronic': '#7bed9f', 'R&B': '#eccc68', 'Latin': '#ff6348',
    'Classical': '#a29bfe', 'Jazz': '#2ed573', 'Country': '#ff9ff3',
    'Metal': '#57606f'
}
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CLUSTER_NAMES = {0: '节奏舞曲型', 1: '舒缓原声型', 2: '高能电子型'}

# ======================== 侧边栏 ========================
with st.sidebar:
    st.markdown("## 🎧 筛选控制台")

    all_genres = sorted(df['Genre'].unique())
    selected_genres = st.multiselect(
        "🎸 流派选择",
        options=all_genres, default=list(all_genres)
    )

    pop_min, pop_max = st.slider(
        "⭐ 流行度区间", 0, 100, (0, 100), 5
    )

    year_min_val, year_max_val = st.slider(
        "📅 发行年份", int(df['Year'].min()), int(df['Year'].max()),
        (int(df['Year'].min()), int(df['Year'].max()))
    )

    explicit_filter = st.radio(
        "🔞 Explicit", ['全部', '仅 Explicit', '仅非 Explicit'], horizontal=True
    )

    st.markdown("---")
    st.caption("筛选条件实时联动所有图表")

# 应用筛选
filtered_df = df[df['Genre'].isin(selected_genres)] if selected_genres else df
filtered_df = filtered_df[(filtered_df['Popularity'] >= pop_min) & (filtered_df['Popularity'] <= pop_max)]
filtered_df = filtered_df[(filtered_df['Year'] >= year_min_val) & (filtered_df['Year'] <= year_max_val)]
if explicit_filter == '仅 Explicit':
    filtered_df = filtered_df[filtered_df['Explicit'] == True]
elif explicit_filter == '仅非 Explicit':
    filtered_df = filtered_df[filtered_df['Explicit'] == False]

# ======================== 标题 ========================
st.title("🎧 Spotify 音乐特征深度分析平台")
st.markdown("---")

# ======================== Tab导航 ========================
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📈 数据概览", "🔬 特征分析", "🎨 关联探索", "🤖 聚类与推荐", "📋 数据明细"
])

# ======================== TAB 0: 数据概览 ========================
with tab0:
    # KPI卡片
    st.subheader("📊 关键指标")
    cols = st.columns(8)
    metrics = [
        ("🎼 总曲目", f"{len(filtered_df):,}"),
        ("⭐ 均流行度", f"{filtered_df['Popularity'].mean():.1f}"),
        ("💃 均舞动性", f"{filtered_df['Danceability'].mean():.3f}"),
        ("⚡ 均能量", f"{filtered_df['Energy'].mean():.3f}"),
        ("😊 均效价", f"{filtered_df['Valence'].mean():.3f}"),
        ("🥁 均BPM", f"{filtered_df['Tempo'].mean():.0f}"),
        ("👤 艺术家", f"{filtered_df['Artist'].nunique()}"),
        ("🔞 Explicit%", f"{filtered_df['Explicit'].mean()*100:.1f}%"),
    ]
    for col, (label, val) in zip(cols, metrics):
        with col:
            st.metric(label, val)

    st.markdown("---")

    # 左：旭日图，右：流行度箱线图
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("**🌐 流派层级分布 — 旭日图**")
        # 构建旭日图数据
        genre_pop = filtered_df.groupby('Genre').agg(
            count=('Track ID', 'count'),
            avg_pop=('Popularity', 'mean')
        ).reset_index()
        genre_pop['level'] = 'All Genres'

        fig_sun = px.sunburst(
            genre_pop,
            path=['level', 'Genre'],
            values='count',
            color='avg_pop',
            color_continuous_scale='Greens',
            labels={'count': '曲目数', 'avg_pop': '平均流行度'},
            hover_data={'count': True, 'avg_pop': ':.1f'}
        )
        fig_sun.update_layout(height=430, margin=dict(t=10, b=10, l=10, r=10))
        fig_sun.update_traces(textinfo='label+percent entry')
        st.plotly_chart(fig_sun, use_container_width=True)

    with col_b:
        st.markdown("**📦 各流派流行度 — 箱线图**")
        fig_box = px.box(
            filtered_df, x='Genre', y='Popularity',
            color='Genre',
            color_discrete_map=GENRE_COLORS,
            labels={'Popularity': '流行度', 'Genre': '流派'},
            points='outliers'
        )
        fig_box.update_layout(
            height=430, template='plotly_dark',
            showlegend=False,
            xaxis_tickangle=-30,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_box.update_traces(marker=dict(size=3, opacity=0.5))
        st.plotly_chart(fig_box, use_container_width=True)

    # 第二行：BPM仪表盘 + 流行度分布
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**🥁 平均BPM — 仪表盘**")
        avg_tempo = filtered_df['Tempo'].mean()
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_tempo,
            number=dict(suffix=" BPM", font=dict(size=40)),
            delta=dict(reference=120, relative=False),
            title=dict(text="当前筛选下平均BPM"),
            gauge=dict(
                axis=dict(range=[50, 200], tickwidth=1),
                bar=dict(color="#1db954"),
                bgcolor="rgba(255,255,255,0.1)",
                steps=[
                    dict(range=[50, 80], color="rgba(169,169,169,0.3)"),
                    dict(range=[80, 110], color="rgba(255,165,2,0.3)"),
                    dict(range=[110, 140], color="rgba(29,185,84,0.3)"),
                    dict(range=[140, 200], color="rgba(255,71,87,0.3)")
                ],
                threshold=dict(line=dict(color="#fff", width=2), value=120)
            )
        ))
        fig_gauge.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_d:
        st.markdown("**⭐ 流行度分级分布 — 漏斗图**")
        pop_dist = filtered_df['Popularity_Level'].value_counts().reindex(
            ['爆款(75-100)', '热门(50-75)', '一般(25-50)', '小众(0-25)']
        )
        fig_funnel = go.Figure(go.Funnel(
            y=pop_dist.index.tolist(),
            x=pop_dist.values.tolist(),
            textposition="inside",
            textinfo="value+percent previous",
            marker=dict(color=['#1db954', '#7bed9f', '#ffa502', '#ff4757'])
        ))
        fig_funnel.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_funnel, use_container_width=True)


# ======================== TAB 1: 特征分析 ========================
with tab1:
    st.subheader("🔬 音频特征多维分析")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**🎻 各流派音频特征分布 — 小提琴图**")
        violin_feat = st.selectbox(
            "选择特征", FEATURES, index=0,
            format_func=lambda f: FEATURE_LABELS.get(f, f),
            key='violin_feat'
        )
        fig_violin = px.violin(
            filtered_df, x='Genre', y=violin_feat,
            color='Genre',
            color_discrete_map=GENRE_COLORS,
            box=True, points='outliers',
            labels={violin_feat: FEATURE_LABELS.get(violin_feat, violin_feat)}
        )
        fig_violin.update_layout(
            height=400, template='plotly_dark',
            showlegend=False, xaxis_tickangle=-30,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_violin.update_traces(marker=dict(size=2, opacity=0.4))
        st.plotly_chart(fig_violin, use_container_width=True)

    with col_b:
        st.markdown("**⫴ 音频特征平行坐标图**")
        parallel_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Instrumentalness', 'Speechiness']
        sample_par = filtered_df.sample(min(500, len(filtered_df)), random_state=2024)

        fig_parallel = go.Figure(go.Parcoords(
            line=dict(
                color=sample_par['Popularity'].values,
                colorscale='Greens',
                showscale=True,
                cmin=0, cmax=100
            ),
            dimensions=[
                dict(label='舞动性', values=sample_par['Danceability'].values, range=[0, 1]),
                dict(label='能量', values=sample_par['Energy'].values, range=[0, 1]),
                dict(label='效价', values=sample_par['Valence'].values, range=[0, 1]),
                dict(label='原声度', values=sample_par['Acousticness'].values, range=[0, 1]),
                dict(label='器乐度', values=sample_par['Instrumentalness'].values, range=[0, 1]),
                dict(label='人声度', values=sample_par['Speechiness'].values, range=[0, 1]),
            ]
        ))
        fig_parallel.update_layout(height=400, margin=dict(t=30, b=10))
        st.plotly_chart(fig_parallel, use_container_width=True)

    # 第二行：相关性热力图 + 流行度相关性条形图
    col_c, col_d = st.columns([3, 2])

    with col_c:
        st.markdown("**🔗 特征相关性矩阵 — 热力图**")
        corr_features = FEATURES + ['Popularity', 'Duration_min']
        corr = filtered_df[corr_features].corr()
        corr_labels = [FEATURE_LABELS.get(f, f) for f in FEATURES] + ['流行度', '时长(分)']

        fig_heat = px.imshow(
            corr, text_auto='.2f', color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1, aspect='auto'
        )
        fig_heat.update_xaxes(ticktext=corr_labels, tickvals=list(range(len(corr_features))))
        fig_heat.update_yaxes(ticktext=corr_labels, tickvals=list(range(len(corr_features))))
        fig_heat.update_layout(height=460, template='plotly_dark',
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_d:
        st.markdown("**📊 流行度相关系数排名**")
        pop_corrs = []
        for feat in FEATURES:
            r = filtered_df[feat].corr(filtered_df['Popularity'])
            pop_corrs.append((FEATURE_LABELS.get(feat, feat), r))
        pop_corrs.sort(key=lambda x: x[1])

        fig_bar = go.Figure(go.Bar(
            x=[r for _, r in pop_corrs],
            y=[n for n, _ in pop_corrs],
            orientation='h',
            marker=dict(
                color=['#ff4757' if r < 0 else '#1db954' for _, r in pop_corrs],
                opacity=0.85
            ),
            text=[f"{r:.4f}" for _, r in pop_corrs],
            textposition='outside'
        ))
        fig_bar.update_layout(
            height=460, template='plotly_dark',
            xaxis_title='Pearson r', yaxis_title='',
            xaxis=dict(range=[-0.5, 0.5]),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ======================== TAB 2: 关联探索 ========================
with tab2:
    st.subheader("🎨 多维关联探索")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**🌌 能量 × 舞动性 × 效价 — 3D散点图**")
        sample_3d = filtered_df.sample(min(1500, len(filtered_df)), random_state=2024)
        fig_3d = px.scatter_3d(
            sample_3d,
            x='Energy', y='Danceability', z='Valence',
            color='Genre',
            size='Popularity',
            size_max=12,
            opacity=0.7,
            color_discrete_map=GENRE_COLORS,
            labels={
                'Energy': '能量', 'Danceability': '舞动性',
                'Valence': '效价', 'Popularity': '流行度'
            },
            hover_data=['Track Name', 'Artist']
        )
        fig_3d.update_layout(
            height=480,
            scene=dict(
                xaxis_title='能量', yaxis_title='舞动性', zaxis_title='效价',
                bgcolor='rgba(0,0,0,0)'
            ),
            legend=dict(orientation='h', y=1.1),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_b:
        st.markdown("**🎤 艺术家影响力 — 气泡图**")
        artist_stats = filtered_df.groupby('Artist').agg(
            Track_Count=('Track ID', 'count'),
            Avg_Popularity=('Popularity', 'mean'),
            Genre_Diversity=('Genre', 'nunique'),
            Followers=('Artist_Followers', 'first')
        ).reset_index()
        artist_stats = artist_stats[artist_stats['Track_Count'] >= 3]  # 至少3首歌
        top_artists = artist_stats.nlargest(50, 'Avg_Popularity')

        fig_bubble = px.scatter(
            top_artists,
            x='Track_Count', y='Avg_Popularity',
            size='Followers',
            size_max=50,
            color='Genre_Diversity',
            color_continuous_scale='Greens',
            hover_data=['Artist', 'Followers'],
            labels={
                'Track_Count': '曲目数', 'Avg_Popularity': '平均流行度',
                'Followers': '粉丝数', 'Genre_Diversity': '流派多样性'
            }
        )
        fig_bubble.update_layout(height=480, paper_bgcolor='rgba(0,0,0,0)')
        fig_bubble.update_traces(opacity=0.8)
        st.plotly_chart(fig_bubble, use_container_width=True)

    # 第二行：调性热力图 + 时长分析
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**🎹 调性 × Mode 流行度热力图**")
        key_mode_data = filtered_df.groupby(['Key', 'Mode']).agg(
            Count=('Track ID', 'count'),
            Avg_Popularity=('Popularity', 'mean')
        ).reset_index()
        key_mode_data['Key_Name'] = key_mode_data['Key'].map(
            {i: KEY_NAMES[i] for i in range(12)}
        )
        key_mode_data['Mode_Name'] = key_mode_data['Mode'].map({1: 'Major', 0: 'Minor'})
        key_mode_data['Label'] = key_mode_data['Key_Name'] + '-' + key_mode_data['Mode_Name']

        pivot = key_mode_data.pivot(index='Key_Name', columns='Mode_Name', values='Avg_Popularity')
        pivot = pivot.reindex(KEY_NAMES)

        fig_km = px.imshow(
            pivot, text_auto='.1f', color_continuous_scale='Greens',
            aspect='auto', labels={'x': '调式', 'y': 'Key', 'color': '均流行度'}
        )
        fig_km.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_km, use_container_width=True)

    with col_d:
        st.markdown("**⏱️ 歌曲时长与流行度关系**")
        dur_bins = [0, 2, 2.5, 3, 3.5, 4, 5, 15]
        dur_labels = ['<2分', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-5', '>5分']
        filtered_df_cp = filtered_df.copy()
        filtered_df_cp['Dur_Bin'] = pd.cut(filtered_df_cp['Duration_min'], bins=dur_bins, labels=dur_labels)

        dur_stats = filtered_df_cp.groupby('Dur_Bin', observed=False).agg(
            Count=('Track ID', 'count'),
            Avg_Pop=('Popularity', 'mean')
        ).reset_index()

        fig_dur = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dur.add_trace(
            go.Bar(x=dur_stats['Dur_Bin'], y=dur_stats['Count'],
                   name='曲目数', marker_color='rgba(29,185,84,0.5)'),
            secondary_y=False
        )
        fig_dur.add_trace(
            go.Scatter(x=dur_stats['Dur_Bin'], y=dur_stats['Avg_Pop'],
                       mode='lines+markers', name='平均流行度',
                       line=dict(color='#ffa502', width=3), marker=dict(size=10)),
            secondary_y=True
        )
        fig_dur.update_layout(height=380, template='plotly_dark',
                              paper_bgcolor='rgba(0,0,0,0)')
        fig_dur.update_yaxes(title_text='曲目数', secondary_y=False)
        fig_dur.update_yaxes(title_text='平均流行度', secondary_y=True)
        st.plotly_chart(fig_dur, use_container_width=True)


# ======================== TAB 3: 聚类与推荐 ========================
with tab3:
    st.subheader("🤖 K-Means 聚类与智能推荐")

    # 准备聚类数据
    cluster_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Instrumentalness', 'Tempo']
    X = filtered_df[cluster_features].copy()
    X['Tempo'] = X['Tempo'] / 250.0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    @st.cache_data
    def do_clustering(data_hash, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=2024, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        return labels, kmeans

    labels, kmeans = do_clustering(hash(str(len(filtered_df))))

    # 手肘法计算（缓存）
    @st.cache_data
    def calc_elbow(data_hash):
        inertias = []
        sil_scores = []
        for k in range(1, 9):
            km = KMeans(n_clusters=k, random_state=2024, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
        # 轮廓系数 (K>=2)
        for k in range(2, 9):
            km = KMeans(n_clusters=k, random_state=2024, n_init=10)
            lbs = km.fit_predict(X_scaled)
            if len(X_scaled) > 5000:
                idx = np.random.choice(len(X_scaled), 5000, replace=False)
                sil = silhouette_score(X_scaled[idx], lbs[idx])
            else:
                sil = silhouette_score(X_scaled, lbs)
            sil_scores.append(sil)
        return inertias, sil_scores

    inertias, sil_scores = calc_elbow(hash(str(len(filtered_df))))

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**📐 手肘法 + 轮廓系数 — 最佳K值**")
        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(
            go.Scatter(x=list(range(1, 9)), y=inertias,
                       mode='lines+markers', name='Inertia (SSE)',
                       line=dict(color='#ff4757', width=2), marker=dict(size=8)),
            secondary_y=False
        )
        fig_elbow.add_trace(
            go.Scatter(x=list(range(2, 9)), y=sil_scores,
                       mode='lines+markers', name='轮廓系数',
                       line=dict(color='#1db954', width=2), marker=dict(size=8)),
            secondary_y=True
        )
        fig_elbow.update_layout(height=380, template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)')
        fig_elbow.update_yaxes(title_text='Inertia (SSE)', secondary_y=False)
        fig_elbow.update_yaxes(title_text='轮廓系数', secondary_y=True, range=[0, 0.5])
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_b:
        st.markdown("**🎨 K-Means 聚类 3D 散点图**")
        sample_cl = filtered_df.sample(min(2000, len(filtered_df)), random_state=2024)
        sample_indices = sample_cl.index
        sample_labels = labels[filtered_df.index.isin(sample_indices)]

        cluster_color_map = {0: '#1db954', 1: '#ffa502', 2: '#a29bfe'}
        fig_cluster3d = px.scatter_3d(
            sample_cl, x='Energy', y='Danceability', z='Valence',
            color=[CLUSTER_NAMES.get(l, f'C{l}') for l in sample_labels],
            color_discrete_map={
                '节奏舞曲型': '#1db954', '舒缓原声型': '#ffa502', '高能电子型': '#a29bfe'
            },
            size='Popularity', size_max=10, opacity=0.65,
            labels={'Energy': '能量', 'Danceability': '舞动性', 'Valence': '效价'},
            hover_data=['Track Name', 'Genre']
        )
        fig_cluster3d.update_layout(
            height=380,
            scene=dict(xaxis_title='能量', yaxis_title='舞动性', zaxis_title='效价',
                       bgcolor='rgba(0,0,0,0)'),
            legend=dict(orientation='h', y=1.1),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cluster3d, use_container_width=True)

    # 聚类特征画像表格
    st.markdown("**📋 聚类特征画像**")
    prof_cols = st.columns(3)
    for i in range(3):
        mask = labels == i
        with prof_cols[i]:
            cluster_name = CLUSTER_NAMES.get(i, f'Cluster {i}')
            color = ['#1db954', '#ffa502', '#a29bfe'][i]
            st.markdown(f"### <span style='color:{color}'>● {cluster_name}</span>",
                        unsafe_allow_html=True)
            n_tracks = mask.sum()
            st.markdown(f"- 曲目数: **{n_tracks}** ({n_tracks/len(filtered_df)*100:.1f}%)")
            st.markdown(f"- 均流行度: **{filtered_df.loc[mask, 'Popularity'].mean():.1f}**")
            st.markdown(f"- 均舞动性: **{filtered_df.loc[mask, 'Danceability'].mean():.3f}**")
            st.markdown(f"- 均能量: **{filtered_df.loc[mask, 'Energy'].mean():.3f}**")
            st.markdown(f"- 均效价: **{filtered_df.loc[mask, 'Valence'].mean():.3f}**")
            st.markdown(f"- 均原声度: **{filtered_df.loc[mask, 'Acousticness'].mean():.3f}**")
            st.markdown(f"- 均器乐度: **{filtered_df.loc[mask, 'Instrumentalness'].mean():.3f}**")
            top_g = filtered_df.loc[mask, 'Genre'].value_counts().index[0]
            st.markdown(f"- 主要流派: **{top_g}**")

    st.markdown("---")

    # 相似歌曲推荐
    st.markdown("**🎯 相似歌曲智能推荐**")
    st.caption("选择一首歌曲，基于音频特征的余弦相似度，在同聚类内推荐最相似的10首")

    rec_col_a, rec_col_b = st.columns([2, 3])

    with rec_col_a:
        # 搜索框
        search_term = st.text_input("🔍 搜索歌曲或艺术家", placeholder="输入关键词...", key='rec_search')
        if search_term:
            candidates = filtered_df[
                filtered_df['Track Name'].str.contains(search_term, case=False, na=False) |
                filtered_df['Artist'].str.contains(search_term, case=False, na=False)
            ].head(30)
        else:
            candidates = filtered_df.nlargest(30, 'Popularity')

        selected_track = st.selectbox(
            "选择歌曲",
            options=candidates['Track ID'].tolist(),
            format_func=lambda tid: f"{filtered_df.loc[filtered_df['Track ID']==tid, 'Track Name'].values[0]} — {filtered_df.loc[filtered_df['Track ID']==tid, 'Artist'].values[0]}",
            key='rec_select'
        )

        if st.button("🎯 查找相似歌曲", type="primary"):
            if selected_track:
                idx = filtered_df[filtered_df['Track ID'] == selected_track].index[0]
                target_cluster = labels[filtered_df.index.get_loc(idx)]
                target_vec = X_scaled[filtered_df.index.get_loc(idx)].reshape(1, -1)

                # 同聚类内
                cluster_mask = labels == target_cluster
                cluster_indices = np.where(cluster_mask)[0]
                cluster_vecs = X_scaled[cluster_mask]

                similarities = 1 - cdist(target_vec, cluster_vecs, metric='cosine')[0]
                top_indices = np.argsort(similarities)[::-1][1:11]

                st.session_state['rec_results'] = []
                for rank, si in enumerate(top_indices, 1):
                    orig_idx = filtered_df.index[cluster_indices[si]]
                    st.session_state['rec_results'].append({
                        'rank': rank,
                        'name': filtered_df.loc[orig_idx, 'Track Name'],
                        'artist': filtered_df.loc[orig_idx, 'Artist'],
                        'genre': filtered_df.loc[orig_idx, 'Genre'],
                        'pop': int(filtered_df.loc[orig_idx, 'Popularity']),
                        'sim': round(float(similarities[si]), 4)
                    })

                st.session_state['query_name'] = filtered_df.loc[idx, 'Track Name']
                st.session_state['query_artist'] = filtered_df.loc[idx, 'Artist']
                st.session_state['query_cluster'] = CLUSTER_NAMES.get(target_cluster, '?')

    with rec_col_b:
        if 'rec_results' in st.session_state and st.session_state['rec_results']:
            st.markdown(f"**查询歌曲**: _{st.session_state['query_name']}_ — {st.session_state['query_artist']}  "
                        f"| 所属聚类: {st.session_state['query_cluster']}")
            st.markdown("**推荐结果:**")
            rec_df = pd.DataFrame(st.session_state['rec_results'])
            rec_df.columns = ['排名', '曲目', '艺术家', '流派', '流行度', '余弦相似度']
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
        else:
            st.info('👈 在左侧选择歌曲并点击"查找相似歌曲"')


# ======================== TAB 4: 数据明细 ========================
with tab4:
    st.subheader("📋 数据明细表")

    search_data = st.text_input("🔍 搜索曲目或艺术家", key='table_search')
    sort_col = st.selectbox("排序字段",
                            ['Popularity', 'Danceability', 'Energy', 'Valence', 'Tempo', 'Duration_min'],
                            format_func=lambda x: FEATURE_LABELS.get(x, x),
                            key='table_sort')
    sort_dir = st.radio("排序方向", ['降序', '升序'], horizontal=True, key='table_dir')

    display_cols = ['Track ID', 'Track Name', 'Artist', 'Album', 'Release Date',
                    'Genre', 'Popularity', 'Explicit', 'Danceability', 'Energy',
                    'Valence', 'Acousticness', 'Instrumentalness', 'Speechiness',
                    'Tempo', 'Duration_min', 'Key_Mode']

    display_df = filtered_df[display_cols].copy()
    if search_data:
        display_df = display_df[
            display_df['Track Name'].str.contains(search_data, case=False, na=False) |
            display_df['Artist'].str.contains(search_data, case=False, na=False)
        ]
    display_df = display_df.sort_values(sort_col, ascending=(sort_dir == '升序'))

    st.dataframe(
        display_df, use_container_width=True, hide_index=True,
        column_config={
            'Release Date': st.column_config.DateColumn('发行日期', format='YYYY-MM-DD'),
            'Popularity': st.column_config.NumberColumn('流行度', format='%d'),
            'Danceability': st.column_config.NumberColumn('舞动性', format='%.3f'),
            'Energy': st.column_config.NumberColumn('能量', format='%.3f'),
            'Valence': st.column_config.NumberColumn('效价', format='%.3f'),
            'Acousticness': st.column_config.NumberColumn('原声度', format='%.3f'),
            'Instrumentalness': st.column_config.NumberColumn('器乐度', format='%.3f'),
            'Speechiness': st.column_config.NumberColumn('人声度', format='%.3f'),
            'Tempo': st.column_config.NumberColumn('BPM', format='%.1f'),
            'Duration_min': st.column_config.NumberColumn('时长(分)', format='%.2f'),
        }
    )
    st.caption(f"共 {len(display_df)} 条记录")

# ======================== 页脚 ========================
st.markdown("---")
st.caption("🎧 Spotify 音乐特征深度分析平台 | 数据可视化课程期末作业 | Powered by Streamlit + Plotly")
