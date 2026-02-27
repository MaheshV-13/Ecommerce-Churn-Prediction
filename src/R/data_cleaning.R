# E-COMMERCE CHURN PREDICTION - DATA CLEANING PIPELINE
# Purpose: Load, clean, validate, and export Online Retail II dataset

# --- Load Dependencies ---

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(lubridate)
  library(stringr)
  library(data.table)
  library(janitor)
  library(here)
  library(yaml)
})

source(here("src", "R", "utils.R"))

log_info("Dependencies loaded successfully")

# --- Load Configuration ---

config <- yaml::read_yaml(here("config", "config.yaml"))

RAW_DATA_PATH <- here(config$paths$raw_data)
INTERIM_DATA_PATH <- here(config$paths$interim_data)
SHEET_NAMES <- config$cleaning$sheets
MIN_UNIT_PRICE <- config$cleaning$min_unit_price

log_info("Configuration loaded from config.yaml")

# --- Step 1: Load Data from Excel Sheets ---
# Combine both yearly sheets into single dataframe with proper data types
# to prevent Excel auto-conversion issues (e.g., "01234" → 1234)

log_info("Loading data from Excel...")

sheet1 <- read_excel(
  path = RAW_DATA_PATH,
  sheet = SHEET_NAMES[1],
  col_types = c("text", "text", "text", "numeric", 
                "date", "numeric", "numeric", "text")
)

sheet2 <- read_excel(
  path = RAW_DATA_PATH,
  sheet = SHEET_NAMES[2],
  col_types = c("text", "text", "text", "numeric", 
                "date", "numeric", "numeric", "text")
)

retail_data <- bind_rows(sheet1, sheet2)
rm(sheet1, sheet2)
gc()

log_info(sprintf("Data loaded: %s rows, %d columns", 
                 format(nrow(retail_data), big.mark = ","), 
                 ncol(retail_data)))

# --- Step 2: Standardize Column Names ---
# Convert to snake_case for consistency across R and Python

retail_data <- retail_data %>% clean_names()

log_info("Column names standardized to snake_case")

# --- Step 3: Sort Chronologically ---
# Explicit time-based ordering before any aggregation ensures consistent
# price selection and proper sequencing for later analysis

retail_data <- retail_data %>%
  arrange(customer_id, stock_code, invoice_date)

log_info("Data sorted chronologically by customer, product, and date")

# --- Step 4: Filter Administrative Stock Codes ---
# Remove non-product transactions (postage, discounts, manual adjustments)
# These distort customer behavior metrics and aren't true product purchases

admin_codes <- c("POST", "D", "M", "BANK CHARGES", "AMAZONFEE", 
                 "DOT", "CRUK", "C2", "S")

before_filter <- nrow(retail_data)

retail_data <- retail_data %>%
  filter(
    !stock_code %in% admin_codes,
    !str_detect(stock_code, "^[A-Z]$"),
    str_length(stock_code) >= 5
  )

admin_removed <- before_filter - nrow(retail_data)
if (admin_removed > 0) {
  log_info(sprintf("Removed %s administrative transactions", 
                   format(admin_removed, big.mark = ",")))
}

# --- Step 5: Remove Invalid Prices ---
# Must happen BEFORE aggregation to ensure weighted average is accurate
# Prices should always be positive (even for returns where quantity is negative)

before_price_filter <- nrow(retail_data)

retail_data <- retail_data %>%
  filter(price >= MIN_UNIT_PRICE)

price_filtered <- before_price_filter - nrow(retail_data)
if (price_filtered > 0) {
  log_info(sprintf("Removed %s rows with invalid prices", 
                   format(price_filtered, big.mark = ",")))
}

# --- Step 6: Aggregate Duplicate Invoice-Product Combinations ---
# Sum quantities for same invoice+stock_code (e.g., item scanned twice)
# Calculate weighted average price: (|q1|*p1 + |q2|*p2) / (|q1| + |q2|)
# CRITICAL: Price must be calculated FIRST before quantity is aggregated
# (dplyr evaluates summarise() arguments sequentially top-to-bottom)

retail_data <- retail_data %>%
  group_by(invoice, stock_code, customer_id) %>%
  summarise(
    # Weighted price - MUST come first to access original quantity vector
    price = sum(abs(quantity) * price) / sum(abs(quantity)),
    # Aggregate quantity - this converts vector to scalar
    quantity = sum(quantity),
    # Metadata fields
    description = first(description[!is.na(description)]),
    invoice_date = first(invoice_date),
    country = first(country),
    .groups = "drop"
  ) %>%
  mutate(description = if_else(is.na(description), "UNKNOWN", description))

log_info("Aggregated duplicate invoice-product combinations")

# --- Step 7: Remove Zero-Quantity Rows ---
# Remove data entry errors where quantity was recorded as zero
# NOTE: This does NOT net purchases and returns (those have different invoices)
# Zero-quantity rows are purely data quality issues from the source system

before_zero <- nrow(retail_data)

retail_data <- retail_data %>%
  filter(quantity != 0)

zero_removed <- before_zero - nrow(retail_data)
if (zero_removed > 0) {
  log_info(sprintf("Removed %s rows with zero quantity (data errors)", 
                   format(zero_removed, big.mark = ",")))
}

# --- Step 8: Remove Transactions Without Customer ID ---
# Critical: CustomerID is required for churn analysis
# Cannot track recency, frequency, or monetary value without customer identity
# These are guest checkouts or system errors

missing_summary <- validate_missing_values(
  retail_data, 
  c("customer_id", "description", "invoice", "quantity", "price")
)

before_na <- nrow(retail_data)

retail_data <- retail_data %>%
  filter(!is.na(customer_id))

na_removed <- before_na - nrow(retail_data)
if (na_removed > 0) {
  log_info(sprintf("Removed %s rows with missing CustomerID (%.1f%%)", 
                   format(na_removed, big.mark = ","),
                   na_removed / before_na * 100))
}

# Validate date range
validate_date_range(retail_data, "invoice_date", 2009, 2011)

# --- Step 9: Create Calculated Fields ---
# Add derived columns for temporal analysis and revenue calculations
# Keep total_amount calculation that works with negative quantities (returns)

retail_data <- retail_data %>%
  mutate(
    total_amount = quantity * price,
    is_return = quantity < 0,
    year = year(invoice_date),
    month = month(invoice_date),
    day = day(invoice_date),
    day_of_week_num = wday(invoice_date),
    day_of_week = case_when(
      day_of_week_num == 1 ~ "Sunday",
      day_of_week_num == 2 ~ "Monday",
      day_of_week_num == 3 ~ "Tuesday",
      day_of_week_num == 4 ~ "Wednesday",
      day_of_week_num == 5 ~ "Thursday",
      day_of_week_num == 6 ~ "Friday",
      day_of_week_num == 7 ~ "Saturday"
    ),
    hour = hour(invoice_date),
    invoice_date_only = as.Date(invoice_date)
  ) %>%
  select(-day_of_week_num)

log_info("Calculated fields added: total_amount, is_return, temporal features")

# --- Step 10: Generate Data Quality Report ---

report_lines <- generate_quality_report(retail_data, "invoice_date")

# Add return statistics
return_stats <- retail_data %>%
  summarise(
    n_returns = sum(is_return),
    pct_returns = round(sum(is_return) / n() * 100, 1),
    return_revenue = sum(total_amount[is_return])
  )

report_lines <- c(
  report_lines[1:(length(report_lines)-1)],
  "",
  sprintf("Return Statistics:"),
  sprintf("  Return Transactions: %s (%.1f%%)", 
          format(return_stats$n_returns, big.mark = ","),
          return_stats$pct_returns),
  sprintf("  Returned Revenue: £%s", 
          format(abs(round(return_stats$return_revenue)), big.mark = ",")),
  report_lines[length(report_lines)]
)

cat(paste(report_lines, collapse = "\n"))

# --- Step 11: Export Cleaned Data ---

if (!dir.exists(INTERIM_DATA_PATH)) {
  dir.create(INTERIM_DATA_PATH, recursive = TRUE)
}

output_file <- file.path(INTERIM_DATA_PATH, "cleaned_retail_data.csv")
fwrite(
  retail_data,
  file = output_file,
  quote = TRUE,
  dateTimeAs = "write.csv"
)

log_info(sprintf("Data exported: %s (%.2f MB)", 
                 output_file,
                 file.info(output_file)$size / 1024^2))

rds_file <- file.path(INTERIM_DATA_PATH, "cleaned_retail_data.rds")
saveRDS(retail_data, file = rds_file)

log_info("Data cleaning pipeline complete - ready for feature engineering")

# =============================================================================
# DATA DICTIONARY (Embedded Documentation)
# =============================================================================
# Column Descriptions:
# - invoice: Transaction identifier (includes "C" prefix for returns)
# - stock_code: Product identifier (5+ char alphanumeric, admin codes removed)
# - customer_id: Unique customer identifier (5-digit integer, NO NAs)
# - quantity: Units in transaction (NEGATIVE for returns, POSITIVE for purchases)
# - price: Unit price in GBP (weighted average if duplicates, always positive)
# - description: Product name (UNKNOWN if missing)
# - invoice_date: Transaction timestamp
# - country: Customer's country of residence
# - total_amount: Revenue for line item (quantity × price, negative for returns)
# - is_return: Boolean flag (TRUE if quantity < 0)
# - year/month/day: Date components extracted from invoice_date
# - day_of_week: Day name (Monday-Sunday)
# - hour: Hour of day (0-23)
# - invoice_date_only: Date without time component
# =============================================================================