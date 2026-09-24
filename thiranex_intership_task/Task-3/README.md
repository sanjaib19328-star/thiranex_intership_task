# Task 3: Exploratory Data Analysis of Hotel Booking Demand

> **Thiranex Internship Task 3 — Analytical Reasoning & Comprehensive EDA**  
> **Author:** Sanjai B  
> **Dataset**: [`hotel_booking.csv`](hotel_booking.csv) (119,390 records, 32 attributes)  
> **Primary Notebook**: [`Hotel_Booking_Demand_EDA.ipynb`](Hotel_Booking_Demand_EDA.ipynb)  
> **Formal EDA Report (Markdown)**: [`EDA_Project_Report.md`](EDA_Project_Report.md)  
> **Formal EDA Report (PDF)**: [`EDA_Project_Report.pdf`](EDA_Project_Report.pdf)  
> **Environment**: Python 3.12 / Pandas / NumPy / Matplotlib / Seaborn / SciPy  

---

## 1. Project Objective & Problem Context

The global hospitality sector operates under persistent uncertainty driven by fluctuating demand, pronounced seasonality, diverse guest demographics, and high cancellation volatility. Unanticipated cancellations distort inventory allocation, disrupt staffing and procurement schedules, and erode revenue per available room (**RevPAR**).

This project executes a **rigorous exploratory data analysis (EDA)** on **119,390 hotel booking records** spanning two distinct property categories: **City Hotel** and **Resort Hotel** properties represented in the Hotel Booking Demand dataset, recorded between July 2015 and August 2017.

### Core Analytical Objectives:
1. **Analyze Demand Segmentation:** Quantify booking volume, market share, and revenue differentials between urban and resort properties.
2. **Decode Temporal & Seasonal Cycles:** Map monthly arrival trends and pricing swings across calendar seasons.
3. **Trace Geographic Demographics:** Identify the principal international and domestic feeder markets.
4. **Evaluate Stay Profiles:** Analyze distribution patterns for weekend vs. weekday stays.
5. **Establish the Lead Time Risk Gradient:** Uncover the mathematical relationship between advance booking lead time and cancellation propensity.
6. **Identify Cancellation Determinants:** Determine the commercial, operational, and behavioral drivers of booking cancellations.
7. **Examine Average Daily Rate (ADR) Dynamics:** Profile pricing elasticity and seasonality across property classes.
8. **Detect Micro-Commitment Markers:** Analyze behavioral indicators (e.g., parking space requests, special requests, room changes) that signal confirmed arrival.
9. **Provide Actionable Revenue Strategies:** Formulate data-driven recommendations for dynamic overbooking, direct channel incentives, and cancellation policy design.

---

## 2. Dataset Architecture & Feature Dictionary

The analysis is performed on the benchmark **Hotel Booking Demand dataset** (`hotel_booking.csv`), consisting of **119,390 booking records** across **32 attributes**.

| Category | Attribute | Data Type | Description & Domain Values |
|---|---|---|---|
| **Target** | `is_canceled` | Binary (0/1) | Booking outcome: `0` = Completed/Check-Out, `1` = Canceled |
| **Property** | `hotel` | Categorical | Property classification: `City Hotel` or `Resort Hotel` |
| **Temporal** | `lead_time` | Integer | Number of days between booking reservation date and arrival date |
| | `arrival_date_year` | Integer | Year of arrival date (2015, 2016, 2017) |
| | `arrival_date_month` | Categorical | Month of arrival date (January – December) |
| | `arrival_date_week_number` | Integer | Week number of the arrival date (1 – 53) |
| | `arrival_date_day_of_month` | Integer | Day of the month of arrival date (1 – 31) |
| **Stay Duration** | `stays_in_weekend_nights` | Integer | Number of weekend nights (Saturday or Sunday) stayed or booked |
| | `stays_in_week_nights` | Integer | Number of weekday nights (Monday – Friday) stayed or booked |
| **Demographics** | `adults` | Integer | Number of adult guests |
| | `children` | Float/Integer | Number of children (imputed default: 0) |
| | `babies` | Integer | Number of babies/infants |
| | `country` | Categorical | Country of origin in ISO 3166-1 alpha-3 format (e.g., PRT, GBR, FRA) |
| **Commercial** | `meal` | Categorical | Meal package: `BB` (Bed & Breakfast), `HB` (Half Board), `FB` (Full Board), `SC` (Self Catering), `Undefined` |
| | `market_segment` | Categorical | Designation: `Online TA`, `Offline TA/TO`, `Groups`, `Direct`, `Corporate`, `Complementary`, `Aviation` |
| | `distribution_channel` | Categorical | Intermediary channel: `TA/TO`, `Direct`, `Corporate`, `GDS`, `Undefined` |
| | `agent` | Float/Categorical | ID of the travel agency making the booking (0 = Direct) |
| | `company` | Float/Categorical | ID of the entity/company paying/booking (0 = Private individual) |
| **Behavioral** | `is_repeated_guest` | Binary (0/1) | Whether the guest has booked previously (`1` = Repeat, `0` = New) |
| | `previous_cancellations` | Integer | Number of prior bookings canceled by the customer |
| | `previous_bookings_not_canceled` | Integer | Number of prior bookings not canceled by the customer |
| | `reserved_room_type` | Categorical | Code of room type requested by customer (A – P) |
| | `assigned_room_type` | Categorical | Code for room type assigned by hotel (A – P) |
| | `booking_changes` | Integer | Number of amendments made to reservation prior to check-in |
| | `deposit_type` | Categorical | Deposit guarantee: `No Deposit`, `Non Refund`, `Refundable` |
| | `days_in_waiting_list` | Integer | Days reservation remained in waiting list before confirmation |
| | `customer_type` | Categorical | Category: `Transient`, `Transient-Party`, `Contract`, `Group` |
| | `required_car_parking_spaces` | Integer | Number of vehicle parking spaces requested by customer |
| | `total_of_special_requests` | Integer | Number of special requests made (e.g., high floor, twin bed, crib) |
| **Financial** | `adr` | Continuous | Average Daily Rate (total lodging revenue divided by total stay nights in €) |
| **Operational** | `reservation_status` | Categorical | Status category: `Check-Out`, `Canceled`, `No-Show` |
| | `reservation_status_date` | Date/String | Date at which the last status was set |

---

## 3. Technologies & Analytical Stack

* **Programming Language:** Python 3.12
* **Data Processing & Manipulation:** `pandas` (v2.2+), `numpy` (v1.26+)
* **Statistical Computation:** `scipy.stats` (descriptive metrics, skewness, kurtosis, IQR calculation)
* **Data Visualization & Aesthetics:** `matplotlib` (v3.8+), `seaborn` (v0.13+)
* **Notebook Execution & Reproduction:** `nbformat`, `nbclient`, `ipykernel`

---

## 4. EDA Methodology & Analytical Architecture

Rather than treating EDA as generic data filtering, this project applies a rigorous 10-phase analytical architecture:

```
[Phase 1: Dataset Ingestion]
        │
        ▼
[Phase 2: Structural Audit & Schema Validation]
        │
        ▼
[Phase 3: Data Hygiene & Integrity Verification]
  ├── Missingness imputation (agent, company, country, children)
  ├── Duplicate evaluation and commercial interpretation
  └── Anomaly exclusion (zero-guest bookings, negative ADR)
        │
        ▼
[Phase 4: Statistical Summaries & Distribution Profiling]
  ├── Parametric & non-parametric central tendencies
  └── Skewness and kurtosis diagnostics
        │
        ▼
[Phase 5: Univariate Analysis]
  ├── Market distribution (City vs. Resort)
  ├── Overall cancellation distribution
  ├── Geographic guest origin mapping (Top 10 countries)
  └── Stay length histograms
        │
        ▼
[Phase 6: Bivariate & Multivariable Analysis]
  ├── Seasonal booking curves across property types
  ├── ADR pricing seasonality and elasticity
  ├── Lead time vs. cancellation probability gradient
  ├── Market segment volatility & channel dependencies
  ├── Deposit policy paradox analysis
  └── Behavioral commitment markers (parking & special requests)
        │
        ▼
[Phase 7: Correlation & Multicollinearity Assessment]
  └── Pearson correlation matrix heatmap
        │
        ▼
[Phase 8: Outlier Detection & Robust Treatment]
  └── IQR boundary auditing (lead time, ADR, stay nights)
        │
        ▼
[Phase 9: Pattern Identification & Behavioral Profiling]
  └── Room assignment change effects & loyalty tracking
        │
        ▼
[Phase 10: Key Insights & Actionable Business Recommendations]
```

---

## 5. Visualizations & Analytical Findings

The project generated **12 dedicated, high-resolution figures** illustrating core hospitality dynamics.

### 5.1 Hotel Market Share & Booking Volume
![Figure 1: Hotel Market Share](images/01_hotel_type_distribution.png)
* **Finding:** **City Hotel** accounts for **66.4%** (79,163 records in clean set) of total reservations, while **Resort Hotel** accounts for **33.6%** (40,045 records).
* **Observation:** The City Hotel captures higher transaction volume across all months, which may reflect differences in capacity, turnover rates, and market segment composition.

---

### 5.2 Property-Level Cancellation Disparity
![Figure 2: Cancellation Rates](images/02_cancellation_rates_by_hotel.png)
* **Finding:** **City Hotel** exhibits a **41.7%** cancellation rate, whereas **Resort Hotel** experiences **27.8%**.
* **Observation:** The observed cancellation-rate difference may reflect differences in booking mix, customer type, distribution channel, lead time, deposit policy, and other property-specific attributes.

---

### 5.3 Monthly Booking Seasonality Across Hotel Types
![Figure 3: Monthly Booking Demand](images/03_monthly_booking_trends.png)
* **Finding:** Both properties reach their highest booking volumes during the summer months (**August** and **July**). City Hotel displays secondary volume peaks in **May** and **October**.
* **Observation:** Both properties exhibit strong summer demand peaks, while City Hotel displays secondary volume increases in spring and autumn.

---

### 5.4 Guest Geographic Feeder Markets
![Figure 4: Top Origin Countries](images/04_top_guest_countries.png)
* **Finding:** Domestic Portuguese guests (**PRT**) represent the single largest market at **40.7%**, followed by the United Kingdom (**10.2%**), France (**8.7%**), Spain (**7.2%**), and Germany (**6.1%**).
* **Observation:** Over **73%** of all bookings originate from Western Europe, indicating heavy geographic concentration and exposure to European economic conditions.

---

### 5.5 Market Segment Volume & Cancellation Risk
![Figure 5: Market Segments](images/05_market_segments_and_channels.png)
* **Finding:** **Online Travel Agencies (OTAs)** dominate volume with **56,402 bookings** (36.7% cancellation rate). The **Groups** segment displays elevated vulnerability with a **61.1%** cancellation rate. **Direct bookings** exhibit the lowest cancellation rate at **15.3%**.
* **Observation:** Group reservations exhibit significantly elevated cancellation rates in this dataset, indicating high contract-level volatility compared to individual direct bookings.

---

### 5.6 Length of Stay Patterns
![Figure 6: Length of Stay Distribution](images/06_length_of_stay_distribution.png)
* **Finding:** Median stay duration is **3 nights** (mean: 3.43 nights). City Hotel stays cluster tightly around **1–3 nights**. Resort Hotel displays secondary peaks at **7 nights** and **14 nights**.
* **Observation:** Stays at the Resort Hotel more frequently follow weekly durations, whereas City Hotel reservations concentrate in short stays.

---

### 5.7 The Lead Time Cancellation Gradient
![Figure 7: Lead Time vs Cancellation](images/07_lead_time_vs_cancellation.png)
* **Finding:** A strict monotonic positive gradient connects lead time to cancellation probability:
  * **0–11 Days (Very Low):** 11.7% cancellation rate
  * **12–49 Days (Low):** 32.1% cancellation rate
  * **50–122 Days (Medium):** 39.5% cancellation rate
  * **123–228 Days (High):** 45.0% cancellation rate
  * **229+ Days (Very High):** 57.3% cancellation rate
* **Observation:** As the advance booking horizon increases, the likelihood of reservation cancellation increases monotonically across all five lead-time quintiles.

---

### 5.8 Average Daily Rate (ADR) Pricing Seasonality
![Figure 8: ADR Seasonality](images/08_adr_monthly_seasonality.png)
* **Finding:** **Resort Hotel** ADR demonstrates pronounced seasonal variation, rising from **€49.53** in November to **€188.52** in August (+280%). In contrast, **City Hotel** ADR varies across a narrower band (€84.09 to €123.30).
* **Observation:** Resort pricing reflects strong summer holiday concentration, whereas City Hotel rates maintain closer alignment throughout the year.

---

### 5.9 The Non-Refundable Deposit Paradox
![Figure 9: Deposit and Customer Types](images/09_deposit_and_customer_types.png)
* **Finding:** Bookings under **Non-Refundable deposits** exhibit a **99.4% cancellation rate** compared to **28.4% for No Deposit**.
* **Observation:** Non-refundable reservations in this dataset are concentrated in specific high-cancellation segments (such as group bookings), which accounts for the counter-intuitive aggregate cancellation rate.

---

### 5.10 Guest Loyalty and Cancellation Precedent
![Figure 10: Repeat Guest Loyalty](images/10_repeated_guests_and_history.png)
* **Finding:** Repeat guests cancel at **14.5%**, compared to **37.8%** for first-time visitors. Guests with 1 prior cancellation cancel at **94.9%**.
* **Observation:** Prior cancellation track record and guest loyalty are strongly associated with future booking outcomes.

---

### 5.11 Behavioral Commitment Markers (Parking & Special Requests)
![Figure 11: Parking and Special Requests](images/11_parking_and_special_requests.png)
* **Finding:**
  * **Car Parking:** In this dataset, across **7,414 bookings** requesting ≥1 parking space, the observed cancellation rate was **0.0%** (0 cancellations recorded).
  * **Special Requests:** Reservations with 0 special requests cancel at **47.7%**. This drops to **22.0%** for 1 request, **17.9%** for 3 requests, and **5.3%** for 5 requests.
* **Observation:** Actions involving explicit logistics planning (such as vehicle parking requests) and customized service requests show a strong inverse association with cancellation probability in the recorded data.

---

### 5.12 Pearson Correlation Matrix
![Figure 12: Correlation Heatmap](images/12_correlation_heatmap.png)
* **Key Correlations:**
  * `lead_time` and `is_canceled`: **r = +0.29** (strongest Pearson correlation among the analyzed numerical variables)
  * `total_of_special_requests` and `is_canceled`: **r = -0.23** (protective association)
  * `required_car_parking_spaces` and `is_canceled`: **r = -0.20** (protective association)
  * `stays_in_week_nights` and `stays_in_weekend_nights`: **r = +0.49** (balanced stay expansion)
  * `adults` and `adr`: **r = +0.26** (occupancy-based rate scaling)

---

## 6. Comprehensive Summary of Key Findings

1. **Volume & Property Mix:** City Hotel captures **66.4%** of bookings but suffers a severe **41.7%** cancellation rate.
2. **Seasonality:** Summer peak (July–August) drives maximum volume and allows Resort Hotel ADR to surge by **280%** (€188.52).
3. **Advance Booking Risk:** Bookings made >7 months in advance cancel at **57.3%**, versus **11.7%** for bookings made within 11 days.
4. **Group Vulnerability:** Group bookings cancel at **61.1%**, representing the highest commercial risk in the portfolio.
5. **The Power of Direct Bookings:** Direct bookings cancel only **15.3%** and avoid third-party distribution commissions.
6. **Behavioral Commitment Markers and Observed Cancellation Patterns:** In this dataset, bookings requesting at least one parking space had a **0.0% observed cancellation rate**, and submitting special requests was associated with substantially lower cancellation rates.
7. **Geographic Concentration:** Over **73%** of guests arrive from Western Europe (Portugal, UK, France, Spain, Germany).

---

## 7. Strategic Business & Revenue Recommendations

| Dimension | Strategic Recommendation | Expected Operational / Revenue Impact |
|---|---|---|
| **Dynamic Overbooking** | Use lead-time-specific cancellation estimates as one input to a statistically validated overbooking model, while incorporating room capacity, historical no-shows, walk costs, and service-level constraints. | Maximizes room capacity utilization without incurring walk/relocation costs. |
| **Wholesale & Group Re-Contracting** | Replace flat non-refundable deposits for groups with **graduated non-refundable milestone payments** (e.g., 25% at 90 days, 50% at 60 days, 100% at 30 days). | Mitigates aggregate group default risks. |
| **Pre-Arrival Engagement Protocol** | Deploy automated pre-arrival emails prompting guests to specify arrival times, request parking, or select room preferences. | Induces micro-commitments that correlate with lower observed cancellation rates. |
| **Direct Channel Loyalty Incentives** | Offer complimentary parking and flexible cancellation policies exclusively for bookings made through the direct hotel website. | Shifts volume away from third-party channels, reducing distribution costs and lowering observed cancellation rates. |
| **Resort Off-Season Yield Strategies** | Develop extended-stay packages, wellness retreats, and corporate offsite packages during off-peak periods. | Helps smooth off-season revenue declines in resort properties. |

---

## 8. Repository Structure

```
Task-3/
├── hotel_booking.csv               # Combined Hotel Booking Demand dataset used for analysis
├── city hotel.csv                  # Original City Hotel source dataset
├── resort hotel.csv                # Original Resort Hotel source dataset
├── Hotel_Booking_Demand_EDA.ipynb  # Fully executed Jupyter Notebook with outputs & markdown
├── README.md                       # Complete project overview & documentation
├── EDA_Project_Report.md           # Formal 13-section analytical report (Markdown)
├── EDA_Project_Report.pdf          # Publication-grade compiled PDF report (A4, print-styled)
├── EDA_Project_Report.html         # Self-contained HTML report with embedded charts
├── generate_plots.py               # Standalone script generating the 12 figures
├── build_notebook.py               # Script compiling & executing the notebook
├── convert_report_to_pdf.py        # Script compiling the PDF from the report
└── images/                         # 12 publication-grade 300 DPI visualizations
    ├── 01_hotel_type_distribution.png
    ├── 02_cancellation_rates_by_hotel.png
    ├── 03_monthly_booking_trends.png
    ├── 04_top_guest_countries.png
    ├── 05_market_segments_and_channels.png
    ├── 06_length_of_stay_distribution.png
    ├── 07_lead_time_vs_cancellation.png
    ├── 08_adr_monthly_seasonality.png
    ├── 09_deposit_and_customer_types.png
    ├── 10_repeated_guests_and_history.png
    ├── 11_parking_and_special_requests.png
    └── 12_correlation_heatmap.png
```

---

## 9. How to Run & Reproduce

### Prerequisites
Ensure Python 3.10+ is installed on your system.

### Step 1: Install Required Dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy nbformat nbclient ipykernel
```

### Step 2: Generate Visualizations
To regenerate the 12 figures in the `images/` directory:
```bash
python generate_plots.py
```

### Step 3: Run the Jupyter Notebook
Open and explore the fully executed notebook interactively:
```bash
jupyter notebook Hotel_Booking_Demand_EDA.ipynb
```
Or execute the automated builder:
```bash
python build_notebook.py
```

---

*This project was completed as part of the **Thiranex Data Science Internship (Task 3)**.*
