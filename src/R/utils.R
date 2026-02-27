# Utility Functions for Data Cleaning
# Purpose: Reusable helper functions for validation and logging

# --- Logging Functions ---

log_info <- function(message) {
  cat(sprintf("[INFO] %s\n", message))
}

log_warn <- function(message) {
  cat(sprintf("[WARN] %s\n", message))
}

log_error <- function(message) {
  cat(sprintf("[ERROR] %s\n", message))
}

# --- Data Validation Functions ---

validate_no_negatives <- function(df, column_name, threshold = 0) {
  # Check if a numeric column contains values below threshold
  # Returns count of violations and logs warning if found
  
  violations <- sum(df[[column_name]] <= threshold, na.rm = TRUE)
  
  if (violations > 0) {
    log_warn(sprintf("%s: %d rows with values <= %d found", 
                     column_name, violations, threshold))
  }
  
  return(violations)
}

validate_date_range <- function(df, date_column, min_year, max_year) {
  # Check if dates fall within expected range
  # Returns count of out-of-range dates
  
  dates <- df[[date_column]]
  out_of_range <- sum(year(dates) < min_year | year(dates) > max_year, 
                      na.rm = TRUE)
  
  if (out_of_range > 0) {
    log_warn(sprintf("%d dates outside %d-%d range", 
                     out_of_range, min_year, max_year))
  }
  
  return(out_of_range)
}

validate_missing_values <- function(df, critical_columns) {
  # Check for missing values in critical columns
  # Returns dataframe with missing value counts and percentages
  
  missing_summary <- df %>%
    summarise(across(all_of(critical_columns), ~sum(is.na(.)))) %>%
    pivot_longer(everything(), names_to = "column", values_to = "missing_count") %>%
    mutate(missing_percent = round(missing_count / nrow(df) * 100, 2)) %>%
    filter(missing_count > 0) %>%
    arrange(desc(missing_count))
  
  if (nrow(missing_summary) > 0) {
    log_warn(sprintf("Missing values detected in %d critical columns", 
                     nrow(missing_summary)))
  }
  
  return(missing_summary)
}

# --- Data Quality Report Generator ---

generate_quality_report <- function(df, date_column = "invoice_date") {
  # Generate comprehensive data quality summary
  # Returns formatted report as character vector
  
  report <- c(
    "\n=============================================================================",
    "DATA QUALITY REPORT",
    "=============================================================================\n",
    sprintf("Dataset Dimensions: %s rows × %d columns", 
            format(nrow(df), big.mark = ","), ncol(df)),
    ""
  )
  
  # Unique entities
  if ("customer_id" %in% colnames(df)) {
    report <- c(report, sprintf("Unique Customers: %s", 
                                format(n_distinct(df$customer_id), big.mark = ",")))
  }
  if ("stock_code" %in% colnames(df)) {
    report <- c(report, sprintf("Unique Products: %s", 
                                format(n_distinct(df$stock_code), big.mark = ",")))
  }
  if ("invoice" %in% colnames(df)) {
    report <- c(report, sprintf("Unique Invoices: %s", 
                                format(n_distinct(df$invoice), big.mark = ",")))
  }
  if ("country" %in% colnames(df)) {
    report <- c(report, sprintf("Countries: %d", n_distinct(df$country)))
  }
  
  # Temporal coverage
  if (date_column %in% colnames(df)) {
    date_range <- as.numeric(difftime(max(df[[date_column]]), 
                                      min(df[[date_column]]), 
                                      units = "days"))
    report <- c(report, "",
                sprintf("Date Range: %s to %s", 
                        min(df[[date_column]]), max(df[[date_column]])),
                sprintf("Duration: %.0f days (%.1f months)", 
                        date_range, date_range / 30))
  }
  
  # Numeric summaries
  if ("quantity" %in% colnames(df) && "price" %in% colnames(df)) {
    total_revenue <- sum(df$quantity * df$price, na.rm = TRUE)
    report <- c(report, "",
                sprintf("Quantity Range: %.0f to %.0f (mean: %.2f)", 
                        min(df$quantity), max(df$quantity), mean(df$quantity)),
                sprintf("Price Range: £%.2f to £%.2f (mean: £%.2f)", 
                        min(df$price), max(df$price), mean(df$price)),
                sprintf("Total Revenue: £%s", 
                        format(round(total_revenue), big.mark = ",")))
  }
  
  # Top countries
  if ("country" %in% colnames(df)) {
    top_countries <- df %>%
      count(country, sort = TRUE) %>%
      head(3) %>%
      mutate(pct = round(n / nrow(df) * 100, 1))
    
    report <- c(report, "", "Top 3 Countries:")
    for (i in 1:nrow(top_countries)) {
      report <- c(report, sprintf("  %s: %s (%.1f%%)", 
                                  top_countries$country[i],
                                  format(top_countries$n[i], big.mark = ","),
                                  top_countries$pct[i]))
    }
  }
  
  report <- c(report, 
              "\n=============================================================================\n")
  
  return(report)
}