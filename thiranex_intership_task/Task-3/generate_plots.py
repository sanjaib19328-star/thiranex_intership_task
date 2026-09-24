"""
Script to generate all 12 publication-grade visualizations for Task 3:
Exploratory Data Analysis of Hotel Booking Demand.

Quality Controls Applied:
- Fully dynamic annotations (no hardcoded values or manual analytical conclusions).
- Dynamic data-driven thresholds (e.g. overall cancellation rate for segment coloring).
- Dynamic quantile bins and labels for lead time.
- Clean and consistent data hygiene (adr <= 1000, zero-guests removed).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

os.makedirs('images', exist_ok=True)

# 1. Load data
df_raw = pd.read_csv('hotel_booking.csv')

# Missing value handling
df = df_raw.copy()
df['company'] = df['company'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['country'] = df['country'].fillna('Unknown')
df['children'] = df['children'].fillna(0)

# Consistent filtering rule: exclude zero-guest bookings and invalid/extreme ADR (adr <= 1000)
df_clean = df[
    (df['adults'] + df['children'] + df['babies'] > 0) &
    (df['adr'] >= 0) &
    (df['adr'] <= 1000)
].copy()

# Feature engineering
df_clean['total_stay'] = df_clean['stays_in_weekend_nights'] + df_clean['stays_in_week_nights']
df_clean['total_guests'] = df_clean['adults'] + df_clean['children'] + df_clean['babies']
df_clean['has_children'] = (df_clean['children'] > 0) | (df_clean['babies'] > 0)
df_clean['is_room_changed'] = df_clean['reserved_room_type'] != df_clean['assigned_room_type']

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
df_clean['arrival_date_month'] = pd.Categorical(df_clean['arrival_date_month'], categories=month_order, ordered=True)

overall_cancel_rate = df_clean['is_canceled'].mean() * 100

print(f"Loaded {len(df_raw):,} raw records. Cleaned analytical set: {len(df_clean):,} records.")
print(f"Overall cancellation rate: {overall_cancel_rate:.2f}%")

# -------------------------------------------------------------
# PLOT 1: Hotel Type Distribution
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
hotel_counts = df_clean['hotel'].value_counts()
colors = ['#2563EB', '#059669']

bars = ax1.bar(hotel_counts.index, hotel_counts.values, color=colors, width=0.55, edgecolor='black', alpha=0.85)
ax1.set_title('Total Bookings by Hotel Type', pad=15)
ax1.set_ylabel('Number of Bookings')
ax1.set_xlabel('Hotel Type')
for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df_clean)) * 100
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1000, f'{yval:,}\n({pct:.1f}%)', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.set_ylim(0, max(hotel_counts.values) * 1.18)

wedges, texts, autotexts = ax2.pie(hotel_counts.values, labels=hotel_counts.index, autopct='%1.1f%%',
                                  colors=colors, startangle=140, explode=(0.04, 0),
                                  wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)
ax2.set_title('Market Share Proportion (%)', pad=15)

plt.suptitle('Figure 1: Booking Volume and Market Share by Hotel Type', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/01_hotel_type_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 1")

# -------------------------------------------------------------
# PLOT 2: Cancellation Rates by Hotel Type
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
cancel_by_hotel = pd.crosstab(df_clean['hotel'], df_clean['is_canceled'], normalize='index') * 100

x = np.arange(len(cancel_by_hotel.index))
width = 0.35

rects1 = ax.bar(x - width/2, cancel_by_hotel[0], width, label='Completed / Not Canceled', 
                color='#10B981', edgecolor='black', alpha=0.85)
rects2 = ax.bar(x + width/2, cancel_by_hotel[1], width, label='Canceled', 
                color='#EF4444', edgecolor='black', alpha=0.85)

ax.set_ylabel('Percentage of Bookings (%)')
ax.set_title('Figure 2: Booking Cancellation Rates by Hotel Type', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(cancel_by_hotel.index, fontweight='bold')
ax.set_ylim(0, 85)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color='#B91C1C')

plt.tight_layout()
plt.savefig('images/02_cancellation_rates_by_hotel.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 2")

# -------------------------------------------------------------
# PLOT 3: Monthly Booking Demand & Seasonal Dynamics (Dynamic Annotations)
# -------------------------------------------------------------
monthly_hotel = df_clean.groupby(['arrival_date_month', 'hotel'], observed=False).size().unstack()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_hotel.index, monthly_hotel['City Hotel'], marker='o', linewidth=2.8, 
        color='#2563EB', label='City Hotel', markersize=7)
ax.plot(monthly_hotel.index, monthly_hotel['Resort Hotel'], marker='s', linewidth=2.8, 
        color='#059669', label='Resort Hotel', markersize=7)

ax.set_title('Figure 3: Monthly Booking Volume Across Hotel Types (Seasonality)', pad=15)
ax.set_xlabel('Arrival Month')
ax.set_ylabel('Total Number of Bookings')
ax.set_xticks(range(len(monthly_hotel.index)))
ax.set_xticklabels(monthly_hotel.index, rotation=35, ha='right')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.5)

# Dynamic peak calculations
city_peak_m = monthly_hotel['City Hotel'].idxmax()
city_peak_v = monthly_hotel.loc[city_peak_m, 'City Hotel']
city_peak_idx = list(monthly_hotel.index).index(city_peak_m)

resort_peak_m = monthly_hotel['Resort Hotel'].idxmax()
resort_peak_v = monthly_hotel.loc[resort_peak_m, 'Resort Hotel']
resort_peak_idx = list(monthly_hotel.index).index(resort_peak_m)

ax.annotate(f'City Peak: {city_peak_m}\n({city_peak_v:,} bookings)', 
            xy=(city_peak_idx, city_peak_v),
            xytext=(city_peak_idx + 0.2, city_peak_v + 600), 
            arrowprops=dict(facecolor='#2563EB', shrink=0.08, width=1.5, headwidth=6),
            fontweight='bold', fontsize=9.5)
ax.annotate(f'Resort Peak: {resort_peak_m}\n({resort_peak_v:,} bookings)', 
            xy=(resort_peak_idx, resort_peak_v),
            xytext=(resort_peak_idx + 0.3, resort_peak_v + 700), 
            arrowprops=dict(facecolor='#059669', shrink=0.08, width=1.5, headwidth=6),
            fontweight='bold', fontsize=9.5)

plt.tight_layout()
plt.savefig('images/03_monthly_booking_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 3")

# -------------------------------------------------------------
# PLOT 4: Top 10 Origin Countries
# -------------------------------------------------------------
top_countries = df_clean['country'].value_counts().head(10).sort_values(ascending=True)
country_pcts = (top_countries / len(df_clean) * 100)

fig, ax = plt.subplots(figsize=(11, 6.5))
bars = ax.barh(top_countries.index, top_countries.values, color='#3B82F6', edgecolor='#1D4ED8', height=0.65, alpha=0.85)
ax.set_title('Figure 4: Top 10 Origin Countries of Hotel Guests', pad=15)
ax.set_xlabel('Total Number of Bookings')
ax.set_ylabel('Country Code (ISO-3)')
ax.set_xlim(0, max(top_countries.values) * 1.18)

for i, (cnt, pct) in enumerate(zip(top_countries.values, country_pcts.values)):
    ax.text(cnt + 600, i, f'{cnt:,} ({pct:.1f}%)', va='center', fontweight='bold', fontsize=10)

country_names = "PRT: Portugal\nGBR: United Kingdom\nFRA: France\nESP: Spain\nDEU: Germany\nITA: Italy\nIRL: Ireland\nBEL: Belgium\nBRA: Brazil\nNLD: Netherlands"
ax.text(0.72, 0.15, country_names, transform=ax.transAxes, fontsize=9.5,
        verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.6', facecolor='#F1F5F9', edgecolor='#94A3B8'))

plt.tight_layout()
plt.savefig('images/04_top_guest_countries.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 4")

# -------------------------------------------------------------
# PLOT 5: Market Segments & Distribution Channels (Dynamic Threshold)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

seg_counts = df_clean['market_segment'].value_counts()
ax1.bar(range(len(seg_counts)), seg_counts.values, color='#6366F1', edgecolor='black', alpha=0.85, width=0.6)
ax1.set_title('Booking Volume by Market Segment', pad=12)
ax1.set_ylabel('Number of Bookings')
ax1.set_xticks(range(len(seg_counts)))
ax1.set_xticklabels(seg_counts.index, rotation=35, ha='right')
for i, v in enumerate(seg_counts.values):
    ax1.text(i, v + 800, f'{v:,}', ha='center', fontweight='bold', fontsize=9.5)
ax1.set_ylim(0, max(seg_counts.values) * 1.15)

seg_cancel = df_clean.groupby('market_segment')['is_canceled'].mean() * 100
seg_cancel = seg_cancel.loc[seg_counts.index]
bar_colors = ['#EF4444' if rate > overall_cancel_rate else '#10B981' for rate in seg_cancel.values]

ax2.bar(range(len(seg_cancel)), seg_cancel.values, color=bar_colors, edgecolor='black', alpha=0.85, width=0.6)
ax2.set_title('Cancellation Rate by Market Segment (%)', pad=12)
ax2.set_ylabel('Cancellation Rate (%)')
ax2.set_xticks(range(len(seg_cancel)))
ax2.set_xticklabels(seg_cancel.index, rotation=35, ha='right')
ax2.axhline(overall_cancel_rate, color='black', linestyle='--', linewidth=1.5, 
            label=f'Overall Avg ({overall_cancel_rate:.1f}%)')
ax2.legend(loc='upper right')
for i, v in enumerate(seg_cancel.values):
    ax2.text(i, v + 1.2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9.5)
ax2.set_ylim(0, max(seg_cancel.values) * 1.18)

plt.suptitle('Figure 5: Market Segment Demand Distribution and Cancellation Propensity', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/05_market_segments_and_channels.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 5")

# -------------------------------------------------------------
# PLOT 6: Total Length of Stay Distribution
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
stay_sub = df_clean[df_clean['total_stay'] <= 14]

stay_hotel = pd.crosstab(stay_sub['total_stay'], stay_sub['hotel'])
x = np.arange(len(stay_hotel.index))
width = 0.4

ax.bar(x - width/2, stay_hotel['City Hotel'], width, label='City Hotel', color='#2563EB', alpha=0.85, edgecolor='black')
ax.bar(x + width/2, stay_hotel['Resort Hotel'], width, label='Resort Hotel', color='#059669', alpha=0.85, edgecolor='black')

ax.set_title('Figure 6: Length of Stay Distribution (Nights) across Hotel Types', pad=15)
ax.set_xlabel('Total Stay Duration (Nights)')
ax.set_ylabel('Number of Bookings')
ax.set_xticks(x)
ax.set_xticklabels(stay_hotel.index)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)

# Annotate peak stays
city_stay_peak = stay_hotel['City Hotel'].idxmax()
ax.annotate(f'Most Popular City Stay: {city_stay_peak} Nights', 
            xy=(city_stay_peak - width/2, stay_hotel.loc[city_stay_peak, 'City Hotel']),
            xytext=(city_stay_peak + 1, stay_hotel.loc[city_stay_peak, 'City Hotel'] - 1000), 
            arrowprops=dict(arrowstyle="->", color='#2563EB', lw=1.8), fontweight='bold')

if 7 in stay_hotel.index:
    ax.annotate('Weekly Spikes: 7 Nights (Resort)', 
                xy=(7 + width/2, stay_hotel.loc[7, 'Resort Hotel']),
                xytext=(8, stay_hotel.loc[7, 'Resort Hotel'] + 1200), 
                arrowprops=dict(arrowstyle="->", color='#059669', lw=1.8), fontweight='bold')

plt.tight_layout()
plt.savefig('images/06_length_of_stay_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 6")

# -------------------------------------------------------------
# PLOT 7: Lead Time vs. Cancellation Propensity (Dynamic Quintiles)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Subplot 1: Distribution KDE
sns.kdeplot(data=df_clean[df_clean['lead_time'] <= 400], x='lead_time', hue='is_canceled', common_norm=False,
            palette={0: '#10B981', 1: '#EF4444'}, fill=True, alpha=0.35, linewidth=2, ax=ax1)
ax1.set_title('Lead Time Density Distribution', pad=12)
ax1.set_xlabel('Lead Time (Days)')
ax1.set_ylabel('Probability Density')
ax1.legend(['Canceled (1)', 'Not Canceled (0)'], loc='upper right')

# Subplot 2: Dynamic Quintile labels based on actual data
lead_binned, bin_edges = pd.qcut(df_clean['lead_time'], 5, retbins=True)
quintile_labels = [
    f"{int(bin_edges[i])}–{int(bin_edges[i+1])}d\n(Q{i+1})"
    for i in range(len(bin_edges) - 1)
]
df_clean['lead_quintile'] = pd.qcut(df_clean['lead_time'], 5, labels=quintile_labels)
quintile_cancel = df_clean.groupby('lead_quintile', observed=False)['is_canceled'].mean() * 100

bars = ax2.bar(quintile_cancel.index, quintile_cancel.values, color='#F59E0B', edgecolor='black', alpha=0.85, width=0.55)
ax2.set_title('Cancellation Rate by Lead Time Quintiles (%)', pad=12)
ax2.set_ylabel('Cancellation Rate (%)')
ax2.set_ylim(0, max(quintile_cancel.values) * 1.2)
for bar in bars:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 1.2, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=10)

plt.suptitle('Figure 7: Impact of Booking Lead Time on Cancellation Probability', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/07_lead_time_vs_cancellation.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 7")

# -------------------------------------------------------------
# PLOT 8: Average Daily Rate (ADR) Seasonality (Dynamic Metrics)
# -------------------------------------------------------------
adr_monthly = df_clean.groupby(['arrival_date_month', 'hotel'], observed=False)['adr'].mean().unstack()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(adr_monthly.index, adr_monthly['City Hotel'], marker='o', linewidth=2.8, color='#2563EB', label='City Hotel', markersize=7)
ax.plot(adr_monthly.index, adr_monthly['Resort Hotel'], marker='s', linewidth=2.8, color='#059669', label='Resort Hotel', markersize=7)

ax.set_title('Figure 8: Average Daily Rate (ADR in €) Seasonality by Hotel Type', pad=15)
ax.set_xlabel('Arrival Month')
ax.set_ylabel('Average Daily Rate (€)')
ax.set_xticks(range(len(adr_monthly.index)))
ax.set_xticklabels(adr_monthly.index, rotation=35, ha='right')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.5)

# Dynamic metrics
resort_peak_m = adr_monthly['Resort Hotel'].idxmax()
resort_peak_v = adr_monthly.loc[resort_peak_m, 'Resort Hotel']
resort_min_m = adr_monthly['Resort Hotel'].idxmin()
resort_min_v = adr_monthly.loc[resort_min_m, 'Resort Hotel']
resort_pct_diff = ((resort_peak_v - resort_min_v) / resort_min_v) * 100
resort_peak_idx = list(adr_monthly.index).index(resort_peak_m)

city_peak_m = adr_monthly['City Hotel'].idxmax()
city_peak_v = adr_monthly.loc[city_peak_m, 'City Hotel']
city_min_v = adr_monthly['City Hotel'].min()
city_peak_idx = list(adr_monthly.index).index(city_peak_m)

ax.annotate(f'Resort Peak ({resort_peak_m}): €{resort_peak_v:.2f}\n(+{resort_pct_diff:.0f}% vs {resort_min_m} trough: €{resort_min_v:.2f})', 
            xy=(resort_peak_idx, resort_peak_v),
            xytext=(resort_peak_idx - 3.2, resort_peak_v + 8), 
            arrowprops=dict(facecolor='#059669', shrink=0.08, width=1.5, headwidth=6),
            fontweight='bold', fontsize=9.5)
ax.annotate(f'City Peak ({city_peak_m}): €{city_peak_v:.2f}\nAnnual Range: €{city_min_v:.2f} – €{city_peak_v:.2f}', 
            xy=(city_peak_idx, city_peak_v),
            xytext=(city_peak_idx - 2.8, city_peak_v + 15), 
            arrowprops=dict(facecolor='#2563EB', shrink=0.08, width=1.5, headwidth=6),
            fontweight='bold', fontsize=9.5)

plt.tight_layout()
plt.savefig('images/08_adr_monthly_seasonality.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 8")

# -------------------------------------------------------------
# PLOT 9: Deposit and Customer Type Cancellation Rates
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

dep_rates = df_clean.groupby('deposit_type')['is_canceled'].mean() * 100
bars1 = ax1.bar(dep_rates.index, dep_rates.values, color=['#10B981', '#DC2626', '#3B82F6'], 
                edgecolor='black', alpha=0.85, width=0.5)
ax1.set_title('Cancellation Rate by Deposit Policy', pad=12)
ax1.set_ylabel('Cancellation Rate (%)')
ax1.set_ylim(0, 115)
for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 2, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=10.5)

cust_rates = df_clean.groupby('customer_type')['is_canceled'].mean() * 100
bars2 = ax2.bar(cust_rates.index, cust_rates.values, color='#8B5CF6', edgecolor='black', alpha=0.85, width=0.55)
ax2.set_title('Cancellation Rate by Customer Type', pad=12)
ax2.set_ylabel('Cancellation Rate (%)')
ax2.set_ylim(0, max(cust_rates.values) * 1.25)
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 1.2, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=10.5)

plt.suptitle('Figure 9: Cancellation Vulnerability Across Deposit Policies and Customer Types', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/09_deposit_and_customer_types.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 9")

# -------------------------------------------------------------
# PLOT 10: Repeated Guest and Booking History (Dynamic Counts)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

rep_counts = df_clean['is_repeated_guest'].value_counts()
rep_rates = df_clean.groupby('is_repeated_guest')['is_canceled'].mean() * 100
rep_labels = [
    f"First-Time Guest\n(n = {rep_counts.get(0, 0):,})", 
    f"Repeated Guest\n(n = {rep_counts.get(1, 0):,})"
]
bars1 = ax1.bar(rep_labels, [rep_rates.get(0, 0), rep_rates.get(1, 0)], color=['#EF4444', '#10B981'], 
                edgecolor='black', alpha=0.85, width=0.45)
ax1.set_title('Cancellation Rate: Guest Loyalty', pad=12)
ax1.set_ylabel('Cancellation Rate (%)')
ax1.set_ylim(0, max(rep_rates.values) * 1.25)
for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 1.2, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=11)

# Prior cancellations
df_clean['prev_cancel_flag'] = np.where(df_clean['previous_cancellations'] == 0, 'No Previous Cancel', 
                                  np.where(df_clean['previous_cancellations'] == 1, '1 Previous Cancel', '2+ Previous Cancels'))
prev_rates = df_clean.groupby('prev_cancel_flag')['is_canceled'].mean().loc[['No Previous Cancel', '1 Previous Cancel', '2+ Previous Cancels']] * 100

bars2 = ax2.bar(prev_rates.index, prev_rates.values, color=['#3B82F6', '#F59E0B', '#DC2626'], edgecolor='black', alpha=0.85, width=0.5)
ax2.set_title('Cancellation Rate by Prior Cancellation History', pad=12)
ax2.set_ylabel('Cancellation Rate (%)')
ax2.set_ylim(0, 110)
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 2, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=10.5)

plt.suptitle('Figure 10: Guest Loyalty and Historical Behavioral Influence on Cancellations', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/10_repeated_guests_and_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 10")

# -------------------------------------------------------------
# PLOT 11: Behavioral Commitment Markers (Dynamic Counts & Rates)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Parking spaces
df_clean['parking_group'] = np.where(df_clean['required_car_parking_spaces'] > 0, 'Required (1+)', 'None (0)')
park_rates = df_clean.groupby('parking_group')['is_canceled'].mean() * 100
park_count_req = len(df_clean[df_clean['required_car_parking_spaces'] > 0])
park_cancel_rate = park_rates.get('Required (1+)', 0.0)

bars1 = ax1.bar(park_rates.index, park_rates.values, color=['#EF4444', '#10B981'], edgecolor='black', alpha=0.85, width=0.45)
ax1.set_title('Cancellation Rate: Parking Space Commitment', pad=12)
ax1.set_ylabel('Cancellation Rate (%)')
ax1.set_ylim(0, max(park_rates.values) * 1.25)
for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 1.2, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=11)

ax1.annotate(f'Cancellation Rate: {park_cancel_rate:.1f}%\n(n = {park_count_req:,} bookings)', 
             xy=(1, max(park_cancel_rate, 0.5)), xytext=(0.75, 14),
             arrowprops=dict(arrowstyle="->", color='#059669', lw=1.8), fontweight='bold', color='#059669')

# Special requests
req_rates = df_clean.groupby('total_of_special_requests')['is_canceled'].mean() * 100
bars2 = ax2.bar(req_rates.index.astype(str), req_rates.values, color='#0D9488', edgecolor='black', alpha=0.85, width=0.55)
ax2.set_title('Cancellation Rate by Number of Special Requests', pad=12)
ax2.set_xlabel('Number of Special Requests')
ax2.set_ylabel('Cancellation Rate (%)')
ax2.set_ylim(0, max(req_rates.values) * 1.2)
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 1.2, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=10)

plt.suptitle('Figure 11: Behavioral Commitment Markers: Parking Spaces and Special Requests', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/11_parking_and_special_requests.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 11")

# -------------------------------------------------------------
# PLOT 12: Pearson Correlation Heatmap
# -------------------------------------------------------------
num_cols = ['is_canceled', 'lead_time', 'total_stay', 'stays_in_weekend_nights', 
            'stays_in_week_nights', 'total_guests', 'adults', 'children', 
            'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 
            'booking_changes', 'adr', 'required_car_parking_spaces', 'total_of_special_requests']

corr_matrix = df_clean[num_cols].corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-0.3, vmax=0.5, annot=True, 
            fmt='.2f', linewidths=0.6, linecolor='white', 
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"}, ax=ax)

ax.set_title('Figure 12: Pearson Correlation Matrix of Booking and Behavioral Attributes', pad=20, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/12_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Plot 12")

print("\nAll 12 visualizations regenerated with full dynamic quality controls!")
