import pandas as pd
import numpy as np
import jax.numpy as jnp
import streamlit as st
import matplotlib.pyplot as plt
import os
import pickle
from monthly_events_by_year import monthly_events_by_year
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "lightweight_mmm")))
from lightweight_mmm import lightweight_mmm, optimize_media, plot, preprocessing, utils

st.set_page_config(layout="centered")
plt.rcParams["figure.figsize"] = (6, 4)  
# --- Fungsi add_event_binaries ---
def add_event_binaries(df, monthly_events_by_year):
    import re
    all_events = set()
    for tahun in monthly_events_by_year:
        for bulan in monthly_events_by_year[tahun]:
            for event in monthly_events_by_year[tahun][bulan]:
                all_events.add(event)
    def norm_colname(event):
        return "is_" + re.sub(r'\W+', '_', event.lower()).strip('_')
    event_map = {}
    for tahun in monthly_events_by_year:
        for bulan in monthly_events_by_year[tahun]:
            for event in monthly_events_by_year[tahun][bulan]:
                event_map.setdefault((int(tahun), int(bulan)), []).append(event)
    for event in all_events:
        colname = norm_colname(event)
        df[colname] = 0
    for idx, row in df.iterrows():
        th = pd.to_datetime(row['monthyear']).year
        bl = pd.to_datetime(row['monthyear']).month
        events = event_map.get((th, bl), [])
        for event in events:
            colname = norm_colname(event)
            df.at[idx, colname] = 1
    return df

# summmary df
def summary_df(mmm):
    from numpyro.diagnostics import summary
    from operator import attrgetter

    # Ambil objek MCMC dari LightweightMMM
    mcmc = mmm._mcmc

    sites = mcmc._states[mcmc._sample_field]
    state_sample_field = attrgetter(mcmc._sample_field)(mcmc._last_state)
    if isinstance(sites, dict) and isinstance(state_sample_field, dict):
        sites = {k: v for k, v in sites.items() if k in state_sample_field}

    stats = summary(sites, prob=0.9)
    import pandas as pd
    import re

    # stats adalah dictionary: key = nama parameter, value = dict statistik
    df_stats = pd.DataFrame(stats).T  # Transpose agar parameter jadi baris
    df_stats = df_stats.reset_index().rename(columns={'index': 'parameter'})

    # Fungsi untuk flatten parameter array jadi satu baris per value
    def flatten_df(df):
        rows = []
        for _, row in df.iterrows():
            param = row['parameter']
            if isinstance(row['mean'], (list, tuple, np.ndarray)):
                for idx, val in enumerate(row['mean']):
                    if isinstance(val, (list, tuple, np.ndarray)):
                        for jdx, val2 in enumerate(val):
                            param_name = f"{param}[{idx},{jdx}]"
                            new_row = row.copy()
                            for col in ['mean', 'std', 'median', '5.0%', '95.0%', 'n_eff', 'r_hat']:
                                new_row[col] = row[col][idx][jdx]
                            new_row['parameter'] = param_name
                            rows.append(new_row)
                    else:
                        param_name = f"{param}[{idx}]"
                        new_row = row.copy()
                        for col in ['mean', 'std', 'median', '5.0%', '95.0%', 'n_eff', 'r_hat']:
                            new_row[col] = row[col][idx]
                        new_row['parameter'] = param_name
                        rows.append(new_row)
            else:
                rows.append(row)
        return pd.DataFrame(rows)

    df_stats_flat = flatten_df(df_stats)
    cols = ['parameter', 'mean', 'std', 'median', '5.0%', '95.0%', 'n_eff', 'r_hat']
    df_stats_flat = df_stats_flat[cols]
    df_stats_flat.reset_index(drop=True)

    channel_names = ['meta', 'tv', 'ucontent', 'youtube']

    # Ambil parameter yang diinginkan saja
    params = ['ad_effect_retention_rate', 'coef_media', 'peak_effect_delay', 'coef_extra_features']

    # Filter parameter yang diinginkan
    df_filtered = df_stats_flat[df_stats_flat['parameter'].str.contains('|'.join(params))].copy()

    # Fungsi untuk mengganti [index] dengan nama channel
    def replace_index_with_channel(param, channel_names):
        import re
        match = re.match(r"(.+)\[(\d+)\]", param)
        if match:
            base, idx = match.groups()
            idx = int(idx)
            if idx < len(channel_names):
                return f"{base}[{channel_names[idx]}]"
        return param

    # Terapkan penggantian nama channel untuk parameter yang perlu
    for base in ['ad_effect_retention_rate', 'coef_media', 'peak_effect_delay']:
        mask = df_filtered['parameter'].str.startswith(base + '[')
        df_filtered.loc[mask, 'parameter'] = df_filtered.loc[mask, 'parameter'].apply(lambda x: replace_index_with_channel(x, channel_names))

    # Untuk gamma_peak_effect_delay jika formatnya [i,j], hanya ambil i saja (atau sesuaikan jika perlu)
    df_filtered['parameter'] = df_filtered['parameter'].str.replace(r'\[(\d+),\d+\]', lambda m: f"[{channel_names[int(m.group(1))]}]", regex=True)

    # Reset index
    df_filtered = df_filtered.reset_index(drop=True)

    df_filtered.drop(columns=['n_eff', 'r_hat'], inplace=True)
    return df_filtered


# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('../dataset/data_clean.csv')
    df = add_event_binaries(df, monthly_events_by_year)
    df["event_count"] = df[[c for c in df.columns if c.startswith("is_")]].sum(axis=1)
    df = df.drop([c for c in df.columns if c.startswith("is_")], axis=1)
    return df

df = load_data()

# --- Sidebar Filter ---
st.sidebar.header("Filter Data")
business_units = df['business_level'].unique()
selected_bu = st.sidebar.selectbox("Pilih Business Unit", business_units)


category = 'business_level'
value = selected_bu

# --- Agregasi & Filter ---
df_agg = df.groupby(['monthyear', category], as_index=False).sum(numeric_only=True)
df_bu = df_agg[df_agg[category] == value].copy()
df_bu.drop(columns=['monthyear', category], inplace=True)

channels = [
    'Spend on meta Amount', 
    'Spend on tv Amount', 
    'Spend on ucontent Amount', 
    'Spend on youtube Amount'
]
channel_names = ["Meta", "TV", "UContent", "YouTube"] 
media_data = df_bu[channels].values
target = df_bu["Sales Amount"].values
extra_features = df_bu[["event_count"]].values
costs = media_data.copy()

split_point = media_data.shape[0] - 12
media_data_test = media_data[split_point:, ...]
extra_features_test = extra_features[split_point:, ...]
SEED = 42

# --- Load Model & Scaler ---
model_file = f"mmm_business_level_{selected_bu}.pkl"
scaler_file = f"scaler_business_level_{selected_bu}.pkl"

if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    st.warning("Model atau scaler untuk BU ini belum tersedia.")
    st.stop()

mmm = utils.load_model(file_path=model_file)
with open(scaler_file, "rb") as f:
    scalers = pickle.load(f)
media_scaler = scalers["media_scaler"]
extra_features_scaler = scalers["extra_features_scaler"]
target_scaler = scalers["target_scaler"]
cost_scaler = scalers["cost_scaler"]

# --- TAMPILKAN SUMMARY DI PALING ATAS ---
st.header("Summary Parameter Model")
df_summary = summary_df(mmm)
st.dataframe(df_summary, use_container_width=True)

# --- Media Contribution & ROI ---
media_contribution, roi_hat = mmm.get_posterior_metrics(target_scaler=target_scaler, cost_scaler=cost_scaler)

# --- Layout 2 Kolom ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Fit")
    fig1 = plot.plot_model_fit(mmm, target_scaler=target_scaler)
    st.pyplot(fig1)
    st.write("")
    st.subheader("Media Contribution Percentage")
    fig2 = plot.plot_bars_media_metrics(metric=media_contribution, metric_name="Media Contribution Percentage",channel_names=channel_names)
    st.pyplot(fig2)
    # Tambahkan spacer agar sejajar dengan kolom 2
    st.write("")  # atau st.empty()

with col2:
    st.subheader("Out-of-sample Model Fit")
    new_predictions = mmm.predict(
        media=media_scaler.transform(media_data_test),
        extra_features=extra_features_scaler.transform(extra_features_test),
        seed=SEED
    )
    fig4 = plot.plot_out_of_sample_model_fit(
        out_of_sample_predictions=new_predictions,
        out_of_sample_target=target_scaler.transform(target[split_point:])
    )
    st.write("")  # atau st.empty()
    st.pyplot(fig4)
    st.write("")  # atau st.empty()

    st.subheader("ROI hat")
    fig3 = plot.plot_bars_media_metrics(metric=roi_hat, metric_name="ROI hat",channel_names=channel_names)
    st.write("")  # atau st.empty()
    st.pyplot(fig3)

    

# --- Response Curves di bawah dua kolom ---
st.subheader("Response Curves")
mmm.media_names = channel_names 
fig5 = plot.plot_response_curves(
    media_mix_model=mmm, target_scaler=target_scaler, seed=SEED
)
st.pyplot(fig5)

# --- Optimization (sendiri di bawah) ---
st.markdown("---")
st.header("Budget Optimization")

# --- Add period and budget input ---
col_period, col_budget = st.columns(2)
with col_period:
    n_time_periods = st.selectbox("Select number of periods (months)", list(range(1, 13)), index=11)
with col_budget:
    monthly_budget = st.number_input(
        "Input monthly budget",
        min_value=0.0,
        value=float(jnp.sum(media_data.mean(axis=0))),
        step=1.0,
        format="%.2f"
    )
    # Format as Rupiah
    def format_rupiah(x):
        return "Rp {:,.2f}".format(x).replace(",", "X").replace(".", ",").replace("X", ".")
    st.markdown(f"**Monthly budget:** {format_rupiah(monthly_budget)}")


# --- TAMPILKAN MEAN HISTORICAL BUDGET PER CHANNEL ---
mean_scaled = mmm.media.mean(axis=0)
mean_unscaled = media_scaler.inverse_transform(mean_scaled.reshape(1, -1))[0]
df_mean_budget = pd.DataFrame({
    "Channel": channel_names,
    "Mean Historical Budget": mean_unscaled
})
df_mean_budget["Mean Historical Budget (Rp)"] = df_mean_budget["Mean Historical Budget"].apply(format_rupiah)
st.subheader("Mean Historical Budget per Channel")
st.dataframe(df_mean_budget[["Channel", "Mean Historical Budget (Rp)"]], use_container_width=True)

# --- INPUT PERSENTASE MIN/MAX UNTUK SETIAP CHANNEL ---
st.subheader("Set Minimum & Maximum Percentage of Mean for Each Channel")
min_pcts = []
max_pcts = []
for i, ch in enumerate(channel_names):
    col1, col2 = st.columns(2)
    with col1:
        min_pct = st.number_input(
            f"Min % of mean for {ch}",
            min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"min_{ch}"
        )
    with col2:
        max_pct = st.number_input(
            f"Max % of mean for {ch}",
            min_value=min_pct, max_value=500.0, value=200.0, step=1.0, key=f"max_{ch}"
        )
    min_pcts.append(min_pct)
    max_pcts.append(max_pct)

# Konversi ke array proporsi (0-1)
bounds_lower_pct = jnp.array([(min_pct/100) for min_pct in min_pcts])
bounds_upper_pct = jnp.array([(max_pct/100) for max_pct in max_pcts])


# --- OPTIMIZATION ---
run_optim = st.button("Run Budget Optimization")
if run_optim:
    prices = jnp.ones(mmm.n_media_channels)
    budget = monthly_budget * n_time_periods

    solution, kpi_without_optim, previous_media_allocation = optimize_media.find_optimal_budgets(
        n_time_periods=n_time_periods,
        media_mix_model=mmm,
        extra_features=extra_features_scaler.transform(extra_features_test)[:n_time_periods],
        budget=budget,
        prices=prices,
        media_scaler=media_scaler,
        target_scaler=target_scaler,
        bounds_lower_pct=bounds_lower_pct,
        bounds_upper_pct=bounds_upper_pct,
        seed=SEED
    )
if run_optim:
    prices = jnp.ones(mmm.n_media_channels)
    # Use user input for budget and period
    budget = monthly_budget * n_time_periods

    solution, kpi_without_optim, previous_media_allocation = optimize_media.find_optimal_budgets(
        n_time_periods=n_time_periods,
        media_mix_model=mmm,
        extra_features=extra_features_scaler.transform(extra_features_test)[:n_time_periods],
        budget=budget,
        prices=prices,
        media_scaler=media_scaler,
        target_scaler=target_scaler,
        seed=SEED
    )

    optimal_budget_allocation = prices * solution.x
    previous_budget_allocation = prices * previous_media_allocation

    # Buat DataFrame ringkas
    df_alloc = pd.DataFrame({
        "Channel": channel_names,
        "Previous Allocation": previous_budget_allocation,
        "Optimal Allocation": optimal_budget_allocation
    })
    df_alloc["Change (%)"] = 100 * (df_alloc["Optimal Allocation"] - df_alloc["Previous Allocation"]) / df_alloc["Previous Allocation"]

    # Format angka agar lebih rapi (misal: Rp dan persen)
    def format_rupiah(x):
        return "Rp {:,.2f}".format(x).replace(",", "X").replace(".", ",").replace("X", ".")
    df_alloc["Previous Allocation"] = df_alloc["Previous Allocation"].apply(format_rupiah)
    df_alloc["Optimal Allocation"] = df_alloc["Optimal Allocation"].apply(format_rupiah)
    df_alloc["Change (%)"] = df_alloc["Change (%)"].apply(lambda x: f"{x:+.2f}%")

    st.subheader("Optimal Budget Allocation Table")
    st.dataframe(df_alloc, use_container_width=True)
    # Tambahkan total di bawah tabel
    total_prev = previous_budget_allocation.sum()
    total_opt = optimal_budget_allocation.sum()
    st.markdown(
        f"""
        <div style="text-align:right">
            <b>Total Previous Allocation:</b> {format_rupiah(total_prev)}<br>
            <b>Total Optimal Allocation:</b> {format_rupiah(total_opt)}
        </div>
        """,
        unsafe_allow_html=True
    )

    fig6 = plot.plot_pre_post_budget_allocation_comparison(
        media_mix_model=mmm,
        kpi_with_optim=solution['fun'],
        kpi_without_optim=kpi_without_optim,
        optimal_buget_allocation=optimal_budget_allocation,
        previous_budget_allocation=previous_budget_allocation,
        figure_size=(10,10)
    )
    st.subheader("Pre vs Post Optimization Comparison")
    st.pyplot(fig6)

    # Tambahkan info kenaikan target variable
    increase_pct = 100 * (solution['fun'] - kpi_without_optim) / kpi_without_optim
    st.markdown(
        f"<div style='text-align:right; font-size:18px;'><b>Increase Target Variable: {increase_pct:+.2f}%</b></div>",
        unsafe_allow_html=True
    )
else:
    st.info("Click the button above to run budget optimization.")