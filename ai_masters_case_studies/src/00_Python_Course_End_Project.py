"""
Python Programming Certification Training
End Project – Travel Aggregator Analysis (MyNextBooking)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Converting to PeriodArray")

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir     = os.path.join(project_root, "data", "00_Python_Course_End_Project")

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
bookings_path = os.path.join(data_dir, "Bookings.csv")
sessions_path = os.path.join(data_dir, "Sessions.csv")

bookings = pd.read_csv(bookings_path)
sessions = pd.read_csv(sessions_path)

# Parse datetime columns
bookings["booking_time"] = pd.to_datetime(bookings["booking_time"], utc=True)
sessions["search_time"]  = pd.to_datetime(sessions["search_time"],  format="mixed", utc=True)


# =============================================================================
# Question 1 – Distinct bookings, sessions, and searches
# =============================================================================

def CountDistinct(bookings_df, sessions_df):
    """
    Returns a dict with the count of distinct bookings, sessions, and searches.
    Bookings are sourced from Bookings.csv; sessions and searches from Sessions.csv.
    """
    distinct_bookings = bookings_df["booking_id"].nunique()
    distinct_sessions = sessions_df["session_id"].nunique()
    distinct_searches = sessions_df["search_id"].nunique()
    return {
        "distinct_bookings": distinct_bookings,
        "distinct_sessions": distinct_sessions,
        "distinct_searches": distinct_searches,
    }


print("=" * 60)
print("QUESTION 1 – Distinct Bookings, Sessions, and Searches")
print("=" * 60)

counts = CountDistinct(bookings, sessions)
print(f"  Distinct bookings : {counts['distinct_bookings']}")
print(f"  Distinct sessions : {counts['distinct_sessions']}")
print(f"  Distinct searches : {counts['distinct_searches']}")


# =============================================================================
# Question 2 – Sessions with more than one booking
# =============================================================================

def SessionsWithMultipleBookings(sessions_df):
    """
    Returns a DataFrame of sessions that are linked to more than one booking,
    along with the booking count per session.
    """
    booked = sessions_df.dropna(subset=["booking_id"])
    counts = booked.groupby("session_id")["booking_id"].nunique()
    multi  = counts[counts > 1].reset_index()
    multi.columns = ["session_id", "booking_count"]
    return multi


print("\n" + "=" * 60)
print("QUESTION 2 – Sessions with More Than One Booking")
print("=" * 60)

multi_booking_sessions = SessionsWithMultipleBookings(sessions)
print(f"  Number of sessions with more than one booking: {len(multi_booking_sessions)}")
if not multi_booking_sessions.empty:
    print(f"\n  Top sessions by booking count:")
    print(multi_booking_sessions.sort_values("booking_count", ascending=False)
          .head(10).to_string(index=False))


# =============================================================================
# Question 3 – Day-of-week with highest bookings + pie chart
# =============================================================================

def BookingsByDayOfWeek(bookings_df, save_dir):
    """
    Counts bookings per day of the week, saves a pie chart, and returns
    the counts Series sorted by weekday order.
    """
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    bookings_df = bookings_df.copy()
    bookings_df["day_of_week"] = bookings_df["booking_time"].dt.day_name()
    counts = bookings_df["day_of_week"].value_counts().reindex(day_order).fillna(0).astype(int)

    # Pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    wedge_props = {"edgecolor": "white", "linewidth": 1.2}
    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=wedge_props,
    )
    ax.set_title("Bookings Distribution by Day of the Week", fontsize=13, fontweight="bold")
    plt.tight_layout()

    pie_path = os.path.join(save_dir, "BookingsByDayOfWeek.png")
    plt.savefig(pie_path, dpi=150)
    plt.close()
    print(f"  Pie chart saved -> 'BookingsByDayOfWeek.png'")
    return counts


print("\n" + "=" * 60)
print("QUESTION 3 – Bookings by Day of the Week")
print("=" * 60)

dow_counts = BookingsByDayOfWeek(bookings, data_dir)
print(f"\n  Bookings per day of the week:")
for day, count in dow_counts.items():
    print(f"    {day:<12s}: {count}")
print(f"\n  Highest booking day: {dow_counts.idxmax()}  ({dow_counts.max()} bookings)")


# =============================================================================
# Question 4 – Total bookings and GBV per service
# =============================================================================

def BookingsAndGBVByService(bookings_df):
    """
    Groups by service_name and returns a DataFrame with the total number of
    bookings and total Gross Booking Value (INR_Amount) for each service.
    """
    summary = (
        bookings_df
        .groupby("service_name")
        .agg(
            total_bookings=("booking_id", "count"),
            total_GBV_INR =("INR_Amount", "sum"),
        )
        .reset_index()
        .sort_values("total_bookings", ascending=False)
    )
    return summary


print("\n" + "=" * 60)
print("QUESTION 4 – Total Bookings and GBV per Service")
print("=" * 60)

service_summary = BookingsAndGBVByService(bookings)
print(f"\n{service_summary.to_string(index=False)}")


# =============================================================================
# Question 5 – Most booked route for repeat customers
# =============================================================================

def MostBookedRouteForRepeatCustomers(bookings_df):
    """
    Filters customers who have made more than one booking, then finds the
    most frequently booked from_city -> to_city route among them.
    Returns the top route and its booking count.
    """
    customer_counts = bookings_df.groupby("customer_id")["booking_id"].count()
    repeat_customers = customer_counts[customer_counts > 1].index
    repeat_df = bookings_df[bookings_df["customer_id"].isin(repeat_customers)].copy()
    repeat_df["route"] = repeat_df["from_city"] + " -> " + repeat_df["to_city"]
    route_counts = repeat_df["route"].value_counts()
    top_route = route_counts.idxmax()
    return top_route, route_counts.max(), route_counts


print("\n" + "=" * 60)
print("QUESTION 5 – Most Booked Route (Repeat Customers)")
print("=" * 60)

top_route, top_count, all_routes = MostBookedRouteForRepeatCustomers(bookings)
print(f"  Most booked route : {top_route}  ({top_count} bookings)")
print(f"\n  Top 5 routes among repeat customers:")
print(all_routes.head(5).to_string())


# =============================================================================
# Question 6 – Top 3 cities where customers book most in advance
# =============================================================================

def TopCitiesBookedInAdvance(bookings_df, min_departures=5, top_n=3):
    """
    For each from_city with at least `min_departures` departures, computes the
    average days_to_departure and returns the top `top_n` cities where customers
    book the furthest in advance.
    """
    city_stats = (
        bookings_df
        .groupby("from_city")
        .agg(departure_count=("booking_id", "count"),
             avg_days_in_advance=("days_to_departure", "mean"))
        .reset_index()
    )
    filtered = city_stats[city_stats["departure_count"] >= min_departures]
    top_cities = (
        filtered
        .sort_values("avg_days_in_advance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return top_cities


print("\n" + "=" * 60)
print("QUESTION 6 – Top 3 Cities (Booked Most in Advance)")
print("=" * 60)

top_advance_cities = TopCitiesBookedInAdvance(bookings, min_departures=5, top_n=3)
print(f"  (Minimum 5 departures required per city)\n")
print(top_advance_cities.to_string(index=False))


# =============================================================================
# Question 7 – Correlation heatmap of numerical columns
# =============================================================================

def CorrelationHeatmap(bookings_df, save_dir):
    """
    Computes the Pearson correlation matrix for all numerical columns in
    bookings_df, plots and saves a heatmap, and returns the column pair
    with the highest absolute correlation (excluding self-correlation).
    """
    num_cols = bookings_df.select_dtypes(include="number")
    corr_matrix = num_cols.corr()

    # Find max off-diagonal correlation
    corr_upper = corr_matrix.where(
        ~pd.DataFrame(
            [[i == j for j in range(len(corr_matrix.columns))]
             for i in range(len(corr_matrix.columns))],
            index=corr_matrix.index,
            columns=corr_matrix.columns,
        )
    )
    abs_corr = corr_upper.abs()
    max_idx  = abs_corr.stack().idxmax()
    max_val  = corr_matrix.loc[max_idx]

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap – Numerical Columns (Bookings)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    heatmap_path = os.path.join(save_dir, "CorrelationHeatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"  Heatmap saved -> 'CorrelationHeatmap.png'")
    return max_idx, max_val


print("\n" + "=" * 60)
print("QUESTION 7 – Numerical Correlation Heatmap")
print("=" * 60)

(col_a, col_b), max_corr = CorrelationHeatmap(bookings, data_dir)
print(f"\n  Pair with highest correlation: '{col_a}'  &  '{col_b}'")
print(f"  Correlation value            : {max_corr:.4f}")


# =============================================================================
# Question 8 – Most used device type per service
# =============================================================================

def MostUsedDevicePerService(bookings_df):
    """
    For each service_name, finds the device_type_used with the highest number
    of bookings. Returns a DataFrame with service_name, top_device, and count.
    """
    device_counts = (
        bookings_df
        .groupby(["service_name", "device_type_used"])["booking_id"]
        .count()
        .reset_index()
        .rename(columns={"booking_id": "booking_count"})
    )
    top_device = (
        device_counts
        .sort_values("booking_count", ascending=False)
        .groupby("service_name")
        .first()
        .reset_index()
        .rename(columns={"device_type_used": "top_device"})
    )
    return top_device


print("\n" + "=" * 60)
print("QUESTION 8 – Most Used Device Type per Service")
print("=" * 60)

device_by_service = MostUsedDevicePerService(bookings)
print(f"\n{device_by_service.to_string(index=False)}")


# =============================================================================
# Question 9 – Quarterly time series of bookings by device type
# =============================================================================

def QuarterlyBookingsByDeviceType(bookings_df, save_dir):
    """
    Aggregates bookings at a quarterly frequency for each device type and
    plots a time series (one line per device type). Saves the chart and
    returns the pivot DataFrame.
    """
    df = bookings_df.copy()
    df["quarter"] = df["booking_time"].dt.to_period("Q").astype(str)

    pivot = (
        df.groupby(["quarter", "device_type_used"])["booking_id"]
        .count()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for device in pivot.columns:
        ax.plot(pivot.index, pivot[device], marker="o", label=device)

    ax.set_title("Quarterly Bookings by Device Type", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year-Quarter")
    ax.set_ylabel("Number of Bookings")
    ax.legend(title="Device Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    ts_path = os.path.join(save_dir, "QuarterlyBookingsByDevice.png")
    plt.savefig(ts_path, dpi=150)
    plt.close()
    print(f"  Time series chart saved -> 'QuarterlyBookingsByDevice.png'")
    return pivot


print("\n" + "=" * 60)
print("QUESTION 9 – Quarterly Bookings by Device Type")
print("=" * 60)

quarterly_pivot = QuarterlyBookingsByDeviceType(bookings, data_dir)
print(f"\n  Quarterly bookings pivot table:")
print(quarterly_pivot.to_string())


# =============================================================================
# Question 10 – Overall Booking-to-Search Ratio (oBSR)
# =============================================================================

def ComputeOBSR(bookings_df, sessions_df):
    """
    Computes the overall Booking-to-Search Ratio (oBSR) per date:
        oBSR(date) = bookings on that date / searches on that date

    Returns a DateFrame indexed by date with columns:
        searches, bookings, oBSR

    Also returns:
        - avg_obsr_by_month   : average oBSR for each month of the year
        - avg_obsr_by_weekday : average oBSR for each day of the week
    """
    # Daily searches from Sessions.csv
    sessions_copy = sessions_df.copy()
    sessions_copy["date"] = sessions_copy["search_time"].dt.date
    daily_searches = sessions_copy.groupby("date")["search_id"].nunique().rename("searches")

    # Daily bookings from Bookings.csv
    bookings_copy = bookings_df.copy()
    bookings_copy["date"] = bookings_copy["booking_time"].dt.date
    daily_bookings = bookings_copy.groupby("date")["booking_id"].nunique().rename("bookings")

    # Merge on date (inner join keeps only dates with both searches and bookings)
    daily = pd.DataFrame({"searches": daily_searches, "bookings": daily_bookings}).fillna(0)
    daily = daily[daily["searches"] > 0].copy()
    daily["oBSR"] = daily["bookings"] / daily["searches"]
    daily.index = pd.to_datetime(daily.index)

    # Average oBSR by month
    avg_by_month = (
        daily.groupby(daily.index.month)["oBSR"]
        .mean()
        .rename_axis("month")
        .reset_index()
    )
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    avg_by_month["month_name"] = avg_by_month["month"].map(month_names)

    # Average oBSR by day of week
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    avg_by_weekday = (
        daily.groupby(daily.index.day_name())["oBSR"]
        .mean()
        .reindex(day_order)
        .rename_axis("day_of_week")
        .reset_index()
    )

    return daily, avg_by_month, avg_by_weekday


def PlotOBSRTimeSeries(daily_obsr, save_dir):
    """Plots and saves the oBSR time series over all available dates."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_obsr.index, daily_obsr["oBSR"], color="#4e79a7", linewidth=1.2)
    ax.set_title("Overall Booking-to-Search Ratio (oBSR) – Daily Time Series",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("oBSR")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(save_dir, "oBSR_TimeSeries.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  oBSR time series saved -> 'oBSR_TimeSeries.png'")


print("\n" + "=" * 60)
print("QUESTION 10 – Overall Booking-to-Search Ratio (oBSR)")
print("=" * 60)

daily_obsr, obsr_by_month, obsr_by_weekday = ComputeOBSR(bookings, sessions)

print(f"\n  Average oBSR per month of the year:")
for _, row in obsr_by_month.iterrows():
    print(f"    {row['month_name']:>3s} : {row['oBSR']:.4f}")

print(f"\n  Average oBSR per day of the week:")
for _, row in obsr_by_weekday.iterrows():
    print(f"    {row['day_of_week']:<12s}: {row['oBSR']:.4f}")

PlotOBSRTimeSeries(daily_obsr, data_dir)

print("\n" + "=" * 60)
print("All questions answered. Charts saved to the data directory.")
print("=" * 60)
