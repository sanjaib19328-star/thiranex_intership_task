# E-Commerce Pricing & Inventory Analysis

## Project Overview
This project explores the pricing strategies and inventory distribution of an e-commerce apparel brand. By analyzing transfer prices (TP), maximum retail prices (MRP), and listing prices across multiple major platforms (Amazon, Flipkart, Myntra, Ajio, etc.), this analysis uncovers how the brand manages profit margins and cross-platform price parity.

## Project Structure
* `May-2022.csv`: The raw, uncleaned dataset containing 1,330 product records.
* `cleaned_data.csv`: The processed dataset (1,221 rows) after coercing data types and handling missing values.
* `Data_Cleaning_Visualization.ipynb`: The core Jupyter Notebook containing all data cleaning, feature engineering, and matplotlib/seaborn visualizations.
* `README.md`: Project documentation and key findings.

## Key Findings

* **Data Integrity:** The original dataset was refined to 1,221 rows by handling missing values. No duplicate rows or SKUs were present.
* **Inventory Distribution:** The **Kurta** category dominates the product catalog with 820 distinct items, followed by Kurta Sets.
* **Premium Categories:** Gowns command the highest average final MRP at **3,151.25 INR**.
* **Marketplace Parity:** The brand maintains strict price parity across all platforms. Listed prices are virtually identical everywhere, with Amazon showing a negligible difference (averaging **2,247.79 INR**).
* **Profitability & Strategy:** The average product markup is **1,662.35 INR**, which translates to a massive **309.91%** markup percentage. This suggests the business employs a high-discount marketing strategy to drive consumer sales while protecting its baseline transfer price.

## How to Run This Project

1. Ensure you have Python installed along with the required data science libraries:
   `pip install pandas numpy matplotlib seaborn jupyter`
2. Clone this repository and navigate to the project folder.
3. Launch Jupyter Notebook from your terminal:
   `jupyter notebook`
4. Open `Data_Cleaning_Visualization.ipynb` and run the cells sequentially to reproduce the analysis and charts.

---
**Author:** Sanjai B.