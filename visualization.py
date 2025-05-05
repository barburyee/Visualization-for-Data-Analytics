# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 13:41:19 2025

@author: Admin
"""
#-------------------- 1. Import required libraries ----------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt


# --------------- 2. Load Dataset. -----------------------------#
file_path = ("college_scorecard_selected_columns.csv")

df = pd.read_csv(file_path, low_memory=False)

# Strip leading/trailing spaces and convert to uppercase for consistency
df.columns = df.columns.str.strip()
print("school feeees")
print(df.columns[df.columns.str.contains("tuition", case=False)])

#PRE-PROCESSING STEPS

# ---------------3. Selection of relevant columns.---------------#

selected_columns = [
    'Admission_rate',
    'Midpoint_of_SAT_scores_at_the_institution__math',
    'In_state_tuition_and_fees',
    'Out_of_state_tuition_and_fees',
    'Average_faculty_salary',
    'Average_cost_of_attendance__academic_year_institutions',
    'Percentage_of_undergraduates_who_receive_a_Pell_Grant',
    'Median_family_income',
    'Control_of_institution',
    'Completion_rate_for_first_time_full_time_target'
]

# Create a new DataFrame with only the above selected columns
df_selected = df[selected_columns]

#-------------- 4. Dataset Preprocessing. -------------------#

# Strip leading/trailing whitespace from all column names
df_selected.columns = df_selected.columns.str.strip()

# Save the selected dataset to a new CSV file
df_selected.to_csv("college_scorecard_selected_columns.csv", index=False)


# Handle Missing or Placeholder Values with NaN
df_selected.replace(['PrivacySuppressed', 'NULL', 'NaN', 'nan', ''], pd.NA, inplace=True)


# Drop rows where more than 30% of the data is missing (optional threshold)
df_selected.dropna(thresh=int(df_selected.shape[1] * 0.7), inplace=True)

# Drop remaining rows with any NaNs (or use fillna() if you'd prefer imputation)
df_selected.dropna(inplace=True)


# Listing the numeric columns
numeric_cols = [
    'Admission_rate',
    'Midpoint_of_SAT_scores_at_the_institution__math',
    'In_state_tuition_and_fees',
    'Out_of_state_tuition_and_fees',
    'Average_faculty_salary',
    'Average_cost_of_attendance__academic_year_institutions',
    'Percentage_of_undergraduates_who_receive_a_Pell_Grant',
    'Median_family_income',
    'Completion_rate_for_first_time_full_time_target'
]


# Convert to numeric types
for col in numeric_cols:
    df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

# Convert 'Control_of_institution' to string (for encoding)
df_selected['Control_of_institution'] = df_selected['Control_of_institution'].astype(str)


# Scale and Normalize

scaler = MinMaxScaler()
df_selected[numeric_cols] = scaler.fit_transform(df_selected[numeric_cols])


# Categorical Encoding

# One-hot encode 'Control_of_institution'

df_selected = pd.get_dummies(df_selected, columns=['Control_of_institution'], drop_first=True)


# Outlier detection and removal

z_scores = np.abs(zscore(df_selected[numeric_cols]))
df_selected = df_selected[(z_scores < 3).all(axis=1)]

# data splitting

# Define features and target
X = df_selected.drop(columns=['Completion_rate_for_first_time_full_time_target'])
y = df_selected['Completion_rate_for_first_time_full_time_target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



corr_matrix = df_selected.corr(numeric_only = True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

#==================================VISUALIZATIONS================================

#================= 1. Correlation Heatmap of Numeric values =====================

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Selected College Variables')
plt.tight_layout()
plt.show()


# ================= 2: Histogram of Admission Rate =================
plt.figure(figsize=(8, 6))
sns.histplot(data=df_selected, x='Admission_rate', bins=20, kde=True, color='skyblue')
plt.title('Distribution of Admission Rates')
plt.xlabel('Admission Rate')
plt.ylabel('Number of Institutions')
plt.grid(True)
plt.tight_layout()
plt.show()


# ================= 3: Boxplot of In-State vs Out-of-State Tuition Fees =================


# Prepare data using actual column names
tuition_df = df_selected[[
    'In_state_tuition_and_fees',
    'Out_of_state_tuition_and_fees'
]].copy()

# Renaming for cleaner plot labels
tuition_df.columns = ['In-State', 'Out-of-State']
tuition_melted = tuition_df.melt(var_name='Tuition Type', value_name='Tuition Fee')

# Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Tuition Type',
            y='Tuition Fee',
            data=tuition_melted, 
            palette='Set2',
            legend= False)
plt.title('In-State vs Out-of-State Tuition Fee Distribution')
plt.ylabel('Tuition Fee (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()

# ================= 4: Scatter Plot of In-State vs Out-of-State Tuition =================

# Cleaning the relevant columns
df["In_state_tuition_and_fees"] = pd.to_numeric(df["In_state_tuition_and_fees"], errors='coerce').replace(-1, pd.NA)
df["Out_of_state_tuition_and_fees"] = pd.to_numeric(df["Out_of_state_tuition_and_fees"], errors='coerce').replace(-1, pd.NA)

# Drop rows with missing values
tuition_df = df[["In_state_tuition_and_fees", "Out_of_state_tuition_and_fees"]].dropna()

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=tuition_df,
    x="In_state_tuition_and_fees",
    y="Out_of_state_tuition_and_fees",
    alpha=0.6
)
plt.title("In-State vs Out-of-State Tuition Fees")
plt.xlabel("In-State Tuition ($)")
plt.ylabel("Out-of-State Tuition ($)")
plt.grid(True)
plt.tight_layout()
plt.show()



# ================= 5: Histogram of Median Family Income of Enrolled Students =================

# Clean the income column
df["Median_family_income"] = pd.to_numeric(df["Median_family_income"], errors='coerce').replace(-1, pd.NA)

# Drop missing values
income_df = df["Median_family_income"].dropna()

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(income_df, bins=40, kde=True, color='teal')
plt.title("Distribution of Median Family Income of Enrolled Students")
plt.xlabel("Median Family Income ($)")
plt.ylabel("Number of Institutions")
plt.grid(True)
plt.tight_layout()
plt.show()


# ================= 6: Scatter Plot of Median Family Income vs Completion Rate =================

# Clean and convert columns
df["Median_family_income"] = pd.to_numeric(df["Median_family_income"], errors='coerce').replace(-1, pd.NA)
df["Completion_rate_for_first_time_full_time_target"] = pd.to_numeric(
    df["Completion_rate_for_first_time_full_time_target"], errors='coerce'
).replace(-1, pd.NA)

# Drop missing values
comp_df = df[["Median_family_income", "Completion_rate_for_first_time_full_time_target"]].dropna()

print(comp_df.shape)
print(comp_df.head())


# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=comp_df,
    x="Median_family_income",
    y="Completion_rate_for_first_time_full_time_target",
    alpha=0.6,
    color='slateblue'
)
plt.title("Median Family Income vs Completion Rate")
plt.xlabel("Median Family Income ($)")
plt.ylabel("Completion Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ================= 7: Bar Plot of Average Faculty Salary by Institution Type =================

# Load dataset
df = pd.read_csv("college_scorecard_selected_columns.csv")

# Map numeric codes to institution types if necessary
control_map = {
    0: "Public",
    1: "Private nonprofit",
    2: "Private for-profit"
}
if df["Control_of_institution"].dtype in ['int64', 'float64']:
    df["Control_of_institution"] = df["Control_of_institution"].map(control_map)

# Clean the salary column
df["Average_faculty_salary"] = pd.to_numeric(df["Average_faculty_salary"], errors='coerce')

# Drop missing values for plotting
salary_df = df[["Control_of_institution", "Average_faculty_salary"]].dropna()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=salary_df,
    x="Control_of_institution",
    y="Average_faculty_salary",
    hue="Control_of_institution",  # now using hue
    palette="Set2",
    legend=False  # turn off duplicate legend
)
plt.title("Average Faculty Salary by Type of Institution", fontsize=14)
plt.xlabel("Institution Type")
plt.ylabel("Average Salary ($)")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


# ================= 8b: Completion Rate vs Median Family Income with Colored Scatter + Linear Regression Line =================

plt.figure(figsize=(10, 6))
sns.lmplot(
    data=df,
    x='Median_family_income',
    y='Completion_rate_for_first_time_full_time_target',
    hue='Control_of_institution',   # e.g., Public, Private nonprofit, etc.
    scatter_kws={'alpha': 0.5},
    line_kws={'linewidth': 2},
    height=6,
    aspect=1.5
)
plt.title('Completion Rate vs Median Family Income by Institution Type')
plt.xlabel('Median Family Income (USD)')
plt.ylabel('Completion Rate')
plt.tight_layout()
plt.show()


# ================= 9: Tuition Fees vs Average Faculty Salary =================

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Out_of_state_tuition_and_fees',
    y='Average_faculty_salary',
    hue='Control_of_institution',
    alpha=0.6
)
plt.title('Faculty Salary vs Out-of-State Tuition by Institution Type')
plt.xlabel('Out-of-State Tuition and Fees (USD)')
plt.ylabel('Average Faculty Salary (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()


# ================= 10: Cost of Attendance vs Admission Rate =================
# Create a dummy 'Control_of_institution' column with sample categories
df['Control_of_institution'] = ['Type A' if i % 3 == 0 else 'Type B' if i % 3 == 1 else 'Type C' for i in range(len(df))]

# Drop rows with missing values
df_clean = df[['Average_cost_of_attendance__academic_year_institutions',
               'Admission_rate',
               'Control_of_institution']].dropna()

# Plot with regression lines and color by the new dummy 'Control_of_institution'
sns.lmplot(
    data=df_clean,
    x='Average_cost_of_attendance__academic_year_institutions',
    y='Admission_rate',
    hue='Control_of_institution',
    scatter_kws={'alpha': 0.5},
    line_kws={'linewidth': 2},
    height=6,
    aspect=1.5
)
plt.title('Admission Rate vs Cost of Attendance by Institution Type')
plt.xlabel('Average Cost of Attendance (USD)')
plt.ylabel('Admission Rate')
plt.tight_layout()
plt.show()
