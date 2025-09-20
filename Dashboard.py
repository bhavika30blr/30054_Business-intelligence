import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import os

# --- 1. Database Connection & Caching ---
# IMPORTANT: Replace with your actual PostgreSQL connection details.
# It is recommended to use environment variables for production.
@st.cache_data
def load_data():
    """Connects to the PostgreSQL database, queries the nonfarm_payrolls table, and caches the data."""
    try:
        conn = psycopg2.connect(
            dbname=os.environ.get("DB_NAME", "ETL_bb"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "Bhavika@12345"),
            host=os.environ.get("DB_HOST", "localhost")
        )
        query = "SELECT * FROM nonfarm_payrolls;"
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Ensure the date column is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        st.success("Data loaded and cached successfully!")
        return df
    except Exception as e:
        st.error(f"Error connecting to the database or loading data: {e}")
        return None

# --- 2. Custom Styling ---
def add_custom_css():
    """Injects custom CSS for styling the app."""
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .css-1av0vzn { /* Streamlit's main header container */
            display: flex;
            justify-content: center;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .st-emotion-cache-1q1n1p { /* CSS for the main content container */
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1), 0 6px 20px 0 rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            background-color: white;
        }
        .css-1f7l053 { /* Plotly chart container */
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.05), 0 6px 20px 0 rgba(0,0,0,0.05);
            padding: 10px;
            background-color: white;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }
        th, td {
            text-align: left;
            padding: 8px;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        </style>
    """, unsafe_allow_html=True)

# --- 3. OLAP Analyses & Visualizations ---
def create_slicing_charts(df):
    """Performs and visualizes Slicing analyses."""
    st.header("Slicing Analysis")

    # Slicing 1: Average payroll employment by year (2010-2025)
    st.subheader("Average Jobs Created (Select Year Range)")
    min_year = int(df['date'].dt.year.min())
    max_year = int(df['date'].dt.year.max())
    year_range = st.slider(
        "Select year range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    df_avg_jobs = df[(df['date'].dt.year >= year_range[0]) & (df['date'].dt.year <= year_range[1])]
    avg_jobs_created = df_avg_jobs['total_nonfarm'].mean()
    st.metric(label=f"Average Jobs Created ({year_range[0]}-{year_range[1]})", value=f"{avg_jobs_created:,.0f}")

    # Sidebar: Display U.S. presidents overlapping the selected year range
    # Static mapping of recent U.S. presidents and their terms (start_year, end_year)
    # Static mapping of recent U.S. presidents and their terms (start_year, end_year)
    # Note: Terms are inclusive years. Adjust or extend as needed.
    presidents_terms = [
        (1993, 2000, 'Bill Clinton'),
        (2001, 2008, 'George W. Bush'),
        (2009, 2016, 'Barack Obama'),
        (2017, 2020, 'Joe Biden'),
        (2021, 2024, 'donald Trump'),  # Assuming current term ends in 2024
    ]

    # Portrait URLs for recent presidents (public domain / Wikimedia). Replace or extend as needed.
    president_portraits = {
        'Bill Clinton': 'https://upload.wikimedia.org/wikipedia/commons/2/20/Bill_Clinton.jpg',
        'George W. Bush': 'https://upload.wikimedia.org/wikipedia/commons/d/d4/George-W-Bush.jpeg',
        'Barack Obama': 'https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg',
        'Donald Trump': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg',
        'Joe Biden': 'https://upload.wikimedia.org/wikipedia/commons/6/68/Joe_Biden_presidential_portrait.jpg'
    }

    def presidents_in_range(start, end):
        """Return a list of presidents whose terms overlap the [start, end] interval."""
        result = []
        for s, e, name in presidents_terms:
            # Check if the terms overlap
            if not (e < start or s > end):
                # Determine overlap span
                overlap_start = max(s, start)
                overlap_end = min(e, end)
                result.append(f"{name} ({overlap_start}-{overlap_end})")
        return result

    # Compute and show presidents in the sidebar as a two-column table (updates when slider changes)
    presidents_rows = []
    for s, e, name in presidents_terms:
        # Check overlap with selected range
        if not (e < year_range[0] or s > year_range[1]):
            overlap_start = max(s, year_range[0])
            overlap_end = min(e, year_range[1])
            presidents_rows.append({'President': name, 'Years': f"{overlap_start}-{overlap_end}"})

    st.sidebar.subheader("U.S. Presidents in Selected Range")
    if presidents_rows:
        pres_df = pd.DataFrame(presidents_rows)

        # Build custom HTML table to include portraits
        rows_html = []
        for _, r in pres_df.iterrows():
            name = r['President']
            years = r['Years']
            img_url = president_portraits.get(name, '')
            img_tag = (
                f"<img src='{img_url}' alt='{name}' style='width:40px;height:40px;border-radius:50%;object-fit:cover;margin-right:8px;'/>"
                if img_url else ""
            )
            rows_html.append(
                f"<tr><td style='vertical-align:middle;padding:6px 8px;'>{img_tag}<strong>{name}</strong></td>"
                f"<td style='vertical-align:middle;padding:6px 8px;color:#03396c;'>{years}</td></tr>"
            )

        table_html = (
            "<table class='pres-table' style='width:100%;border-collapse:collapse;font-family:Arial,sans-serif;'>"
            "<thead>"
            "<tr>"
            "<th style='background:#2563EB;color:white;padding:8px 10px;text-align:left;'>President</th>"
            "<th style='background:#2563EB;color:white;padding:8px 10px;text-align:left;'>Years</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            + "".join(rows_html)
            + "</tbody></table>"
            + "<style>.pres-table tbody tr:hover td { background: #eef6ff; }</style>"
        )
        st.sidebar.markdown(table_html, unsafe_allow_html=True)
    else:
        st.sidebar.info("No presidents found in the selected range.")

    # Slicing 2: Monthly employment comparison for Mar-Dec 2020 vs. 2019
    st.subheader("Monthly Employment Comparison (Mar-Dec 2020 vs. 2019)")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df_slice2 = df[((df['year'] == 2019) | (df['year'] == 2020)) & 
                   (df['month'].between(3, 12))]
    fig2 = px.line(df_slice2, x='date', y='total_nonfarm', color='year',
                   title="Monthly Employment: March-December 2020 vs. 2019",
                   labels={'total_nonfarm': 'Total Employment (in thousands)', 'date': 'Date'})
    st.plotly_chart(fig2)

    # Slicing 2b: Monthly employment distribution (pie) for the selected year range
    st.subheader("Monthly Employment Distribution (Pie Chart)")

    # Filter dataset to the selected year range
    df_monthly_range = df[(df['date'].dt.year >= year_range[0]) & (df['date'].dt.year <= year_range[1])].copy()
    if not df_monthly_range.empty:
        # Aggregate average employment by month across the selected years
        df_monthly_range['month_num'] = df_monthly_range['date'].dt.month
        df_monthly_range['month_name'] = df_monthly_range['date'].dt.strftime('%b')

        monthly_avg = (
            df_monthly_range.groupby(['month_num', 'month_name'])['total_nonfarm']
            .mean()
            .reset_index()
            .sort_values('month_num')
        )

        fig_pie = px.pie(
            monthly_avg,
            names='month_name',
            values='total_nonfarm',
            title=f"Average Monthly Employment Share ({year_range[0]}-{year_range[1]})",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No monthly data available for the selected year range to build the pie chart.")

    # (removed earlier duplicate pre/post-COVID block)
    # Monthly Employment Distribution: Pre-COVID vs COVID-Period vs Post-COVID (Table)
    st.subheader("Monthly Employment: Pre-COVID vs COVID-Period vs Post-COVID (Table)")

    # Define date windows
    pre_cutoff = pd.to_datetime("2020-03-01")
    covid_start = pd.to_datetime("2020-03-01")
    covid_end = pd.to_datetime("2021-12-31")

    pre_df = df[df['date'] < pre_cutoff].copy()
    covid_df = df[(df['date'] >= covid_start) & (df['date'] <= covid_end)].copy()
    post_df = df[df['date'] > covid_end].copy()

    if pre_df.empty and covid_df.empty and post_df.empty:
        st.info("No data available to compare pre-, during-, and post-COVID monthly distributions.")
    else:
        def monthly_avg(seg):
            tmp = seg.copy()
            tmp['month_num'] = tmp['date'].dt.month
            tmp['month_name'] = tmp['date'].dt.strftime('%b')
            return tmp.groupby(['month_num', 'month_name'])['total_nonfarm'].mean().reset_index()

        months_index = pd.DataFrame({'month_num': list(range(1, 13))})
        months_index['month_name'] = months_index['month_num'].apply(lambda m: pd.to_datetime(str(m), format='%m').strftime('%b'))

        pre_monthly = monthly_avg(pre_df) if not pre_df.empty else pd.DataFrame(columns=['month_num', 'month_name', 'total_nonfarm'])
        covid_monthly = monthly_avg(covid_df) if not covid_df.empty else pd.DataFrame(columns=['month_num', 'month_name', 'total_nonfarm'])
        post_monthly = monthly_avg(post_df) if not post_df.empty else pd.DataFrame(columns=['month_num', 'month_name', 'total_nonfarm'])

        pre_monthly = pre_monthly.rename(columns={'total_nonfarm': 'total_nonfarm_pre'})
        covid_monthly = covid_monthly.rename(columns={'total_nonfarm': 'total_nonfarm_cov'})
        post_monthly = post_monthly.rename(columns={'total_nonfarm': 'total_nonfarm_post'})

        merged = months_index.merge(pre_monthly, on=['month_num', 'month_name'], how='left')
        merged = merged.merge(covid_monthly, on=['month_num', 'month_name'], how='left')
        merged = merged.merge(post_monthly, on=['month_num', 'month_name'], how='left')

        # Safe calculations
        merged['Difference'] = merged['total_nonfarm_post'] - merged['total_nonfarm_pre']
        def safe_pct(row):
            pre = row['total_nonfarm_pre']
            diff = row['Difference']
            if pd.isnull(pre) or pre == 0:
                return None
            return (diff / pre) * 100
        merged['Pct Change'] = merged.apply(safe_pct, axis=1)

        display_df = merged[['month_name', 'total_nonfarm_pre', 'total_nonfarm_cov', 'total_nonfarm_post', 'Difference', 'Pct Change']].copy()
        display_df.columns = ['Month', 'Avg Pre-COVID', 'Avg COVID Period', 'Avg Post-COVID', 'Difference', 'Pct Change (%)']

        for col in ['Avg Pre-COVID', 'Avg COVID Period', 'Avg Post-COVID', 'Difference']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.1f}" if pd.notnull(x) else 'N/A')
        display_df['Pct Change (%)'] = display_df['Pct Change (%)'].apply(lambda x: f"{x:,.1f}%" if pd.notnull(x) and not pd.isna(x) else 'N/A')

        table_html = display_df.to_html(index=False, classes='cmp-table')
        st.markdown(
            """
            <style>
            .cmp-table { width:100%; border-collapse: collapse; font-family: Arial, sans-serif; }
            .cmp-table thead th { background: #059669; color: white; padding: 8px 10px; text-align: left; font-weight: 600; }
            .cmp-table tbody td { padding: 8px 10px; border-bottom: 1px solid #e6f3ee; color: #014f3b; }
            .cmp-table tbody tr:hover td { background: #ecfdf5; }
            .cmp-table tbody td:nth-child(5) { font-weight: 600; } /* highlight Difference column */
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(table_html, unsafe_allow_html=True)

def create_dicing_charts(df):
    """Performs and visualizes Dicing analyses."""
    st.header("Dicing Analysis")

    # Dicing 1: Months with > 2% month-over-month employment drop
    st.subheader("Months with > 2% Month-over-Month Employment Drop")
    df['mom_growth'] = df['total_nonfarm'].pct_change() * 100
    df['month_year'] = df['date'].dt.strftime('%b-%Y')
    
    significant_drops = df[df['mom_growth'] < -2].copy()
    if not significant_drops.empty:
        st.write("Months with a greater than 2% month-over-month employment drop:")
        st.dataframe(significant_drops[['month_year', 'mom_growth']].round(2).rename(columns={'mom_growth': 'MoM Growth (%)'}))
        
        # Calculate recovery time
        recovery_data = []
        for index, row in significant_drops.iterrows():
            drop_date = row['date']
            drop_employment = row['total_nonfarm']
            
            # Find the peak before the drop
            pre_drop_data = df[df['date'] < drop_date]
            if not pre_drop_data.empty:
                prior_peak_employment = pre_drop_data['total_nonfarm'].max()
                
                # Find the first month where employment recovers to or exceeds the prior peak
                post_drop_data = df[df['date'] > drop_date]
                recovery_month = post_drop_data[post_drop_data['total_nonfarm'] >= prior_peak_employment].first_valid_index()
                
                if recovery_month:
                    months_to_recover = (df.loc[recovery_month]['date'].year - drop_date.year) * 12 + (df.loc[recovery_month]['date'].month - drop_date.month)
                    recovery_data.append({
                        'Drop Month': row['month_year'],
                        'Prior Peak Date': df.loc[pre_drop_data['total_nonfarm'].idxmax()]['date'].strftime('%b-%Y'),
                        'Months to Recover': months_to_recover
                    })
                else:
                    recovery_data.append({'Drop Month': row['month_year'], 'Prior Peak Date': 'N/A', 'Months to Recover': 'Not recovered yet'})
        
                if recovery_data:
                        st.write("Time taken to recover to the prior peak:")
                        rec_df = pd.DataFrame(recovery_data)

                        # Build a colorful HTML table with bold fonts to make it pop
                        def _format_cell(val):
                                return str(val)

                        # Header and style
                        table_html = """
                        <style>
                        .rec-table { width:100%; border-collapse: collapse; font-family: Arial, sans-serif; }
                        .rec-table thead th { background: linear-gradient(90deg, #f97316, #f59e0b); color: white; padding: 10px; text-align: left; font-weight: 800; }
                        .rec-table tbody td { padding: 10px; border-bottom: 1px solid #f3f4f6; font-weight:700; color: #b91c1c; }
                        .rec-table tbody tr:nth-child(odd) { background: #fff7ed; }
                        .rec-table tbody tr:nth-child(even) { background: #fffaf0; }
                        .rec-table tbody tr:hover td { background: #fff1d6; }
                        .badge-notrecovered { color: #b91c1c; font-weight:800; }
                        </style>
                        <table class='rec-table'>
                            <thead>
                                <tr>
                                    <th>Drop Month</th>
                                    <th>Prior Peak Date</th>
                                    <th>Months to Recover</th>
                                </tr>
                            </thead>
                            <tbody>
                        """

                        for _, r in rec_df.iterrows():
                                months = r['Months to Recover']
                                if isinstance(months, str) and 'Not recovered' in months:
                                        months_html = f"<span class='badge-notrecovered'>{months}</span>"
                                else:
                                        months_html = f"{months}"
                                table_html += f"<tr><td>{_format_cell(r['Drop Month'])}</td><td>{_format_cell(r['Prior Peak Date'])}</td><td>{months_html}</td></tr>"

                        table_html += "</tbody></table>"
                        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No months found with a month-over-month employment drop greater than 2%.")

    # Dicing 2: Quarterly payroll growth trends
    st.subheader("Quarterly Payroll Growth Trends by Month")
    # Calculate month-over-month percentage change for all months
    df_all = df.copy()
    df_all['year'] = df_all['date'].dt.year
    df_all['month'] = df_all['date'].dt.strftime('%b')
    df_all['month_num'] = df_all['date'].dt.month
    df_all['pct_change_mom'] = df_all['total_nonfarm'].pct_change() * 100

    # Quarter selection dropdown
    quarter_map = {
        'Q1': [1, 2, 3],
        'Q2': [4, 5, 6],
        'Q3': [7, 8, 9],
        'Q4': [10, 11, 12]
    }
    quarter = st.selectbox("Select Quarter for Analysis:", list(quarter_map.keys()), index=3)
    selected_months = quarter_map[quarter]

    # Filter for selected quarter months only
    df_quarter = df_all[df_all['month_num'].isin(selected_months)].copy()

    # Year slider
    min_year = int(df_quarter['year'].min())
    max_year = int(df_quarter['year'].max())
    year_range = st.slider(
        f"Select year range for {quarter} analysis:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    df_quarter_interval = df_quarter[(df_quarter['year'] >= year_range[0]) & (df_quarter['year'] <= year_range[1])]

    # Custom color mapping for months
    color_map = {
        'Q1': {'Jan': 'red', 'Feb': 'blue', 'Mar': 'green'},
        'Q2': {'Apr': 'red', 'May': 'blue', 'Jun': 'green'},
        'Q3': {'Jul': 'red', 'Aug': 'blue', 'Sep': 'green'},
        'Q4': {'Oct': 'red', 'Nov': 'blue', 'Dec': 'green'}
    }
    month_color_map = color_map[quarter]

    # Line chart: one line per month in selected quarter
    fig3 = px.line(
        df_quarter_interval,
        x='year',
        y='pct_change_mom',
        color='month',
        labels={'year': 'Year', 'pct_change_mom': 'MoM % Change', 'month': 'Month'},
        color_discrete_map=month_color_map,
        markers=True,
        title=f"{quarter} Payroll Growth Trends by Month"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Quarter Key Events (dynamic, per selected year range and quarter)
    st.subheader("Quarter Key Events")

    # Minimal event mapping 2010-2025. Extend as needed.
    # Keys are tuples (year, quarter_label)
    quarter_events = {
        (2010, 'Q1'): 'Recovery from 2008-09 recession continues; payrolls slowly improve.',
        (2010, 'Q2'): 'Small but steady job gains across services sectors.',
        (2010, 'Q3'): 'Hiring increases in healthcare and education.',
        (2010, 'Q4'): 'Consumer demand strengthens ahead of holidays.',
        (2011, 'Q1'): 'Manufacturing shows modest stabilization.',
        (2012, 'Q1'): 'Jobs grow; policy uncertainty affects hiring in some sectors.',
        (2013, 'Q1'): 'Broad-based hiring in professional services.',
        (2014, 'Q1'): 'Tech sector leads in payroll contributions.',
        (2015, 'Q1'): 'Labor market tightness emerges in some metros.',
        (2016, 'Q1'): 'Slow growth with sector rotation into healthcare.',
        (2017, 'Q1'): 'Tax reform discussions influence business hiring plans.',
        (2018, 'Q1'): 'Stronger wage growth appears in lower-wage sectors.',
        (2019, 'Q1'): 'Stable hiring; unemployment near multi-year lows.',
        (2020, 'Q1'): 'COVID-19 shock begins; sharp job losses in March.',
        (2020, 'Q2'): 'Widespread shutdowns; unprecedented employment declines.',
        (2020, 'Q3'): 'Partial reopening, strong but uneven recovery.',
        (2020, 'Q4'): 'Recovery continues with stimulus support and policy shifts.',
        (2021, 'Q1'): 'Vaccination rollout begins; labor markets start adjusting.',
        (2021, 'Q2'): 'Hiring accelerates as reopenings expand.',
        (2021, 'Q3'): 'Labor shortages in leisure & hospitality persist.',
        (2021, 'Q4'): 'Robust job gains across many sectors.',
        (2022, 'Q1'): 'Strong post-pandemic recovery continues.',
        (2022, 'Q2'): 'Wage pressures and inflation concerns grow.',
        (2022, 'Q3'): 'Labor market remains tight; turnover elevated.',
        (2022, 'Q4'): 'Cooling begins in certain tech and finance roles.',
        (2023, 'Q1'): 'Slower hiring as monetary policy tightens.',
        (2023, 'Q2'): 'Selective weakness in interest-rate-sensitive sectors.',
        (2023, 'Q3'): 'Stabilization with pockets of hiring strength.',
        (2023, 'Q4'): 'Moderate gains; employers adjust to higher rates.',
        (2024, 'Q1'): 'Election year dynamics influence business sentiment.',
        (2024, 'Q2'): 'Gradual moderation in hiring activity.',
        (2024, 'Q3'): 'Sectoral shifts as firms rebalance hiring.',
        (2024, 'Q4'): 'Year-end hiring topped by services and healthcare.',
        (2025, 'Q1'): 'Early signals for 2025 hiring plans emerge.'
    }

    # Build a list of years in the year_range slider selection
    years = list(range(year_range[0], year_range[1] + 1))

    # Filter to only the years that exist in the dataset interval for this quarter selection
    display_events = []
    for y in years:
        key = (y, quarter)
        text = quarter_events.get(key, 'No notable event recorded for this quarter.')
        display_events.append((y, text))

    # Render events with bold black font on light-yellow background
    events_html_rows = []
    for y, txt in display_events:
        events_html_rows.append(
            f"""<div style="background:#FFF9C4;padding:12px;border-radius:8px;margin-bottom:8px;"> 
            <strong style="color:black;font-weight:700;">Q{quarter[-1]} {y}:</strong>
            <span style="color:black;margin-left:8px;">{txt}</span>
            </div>"""
        )

    st.markdown("".join(events_html_rows), unsafe_allow_html=True)

    # Q4 analysis: For each year, which month in Q4 (Oct-Dec) shows the highest MoM payroll growth?
    st.subheader("Q4: Month with Highest Payroll Growth per Year")
    # Ensure pct_change_mom exists on df_all (computed earlier)
    if 'pct_change_mom' not in df_all.columns:
        df_all['pct_change_mom'] = df_all['total_nonfarm'].pct_change() * 100

    # Filter to Q4 months only
    df_q4 = df_all[df_all['month_num'].isin([10, 11, 12])].copy()
    if df_q4.empty:
        st.info("No Q4 data available for the selected interval.")
    else:
        # For each year, find the month with the maximum pct_change_mom
        q4_top = df_q4.loc[df_q4.groupby('year')['pct_change_mom'].idxmax()].copy()
        # Some years may have NaN pct_change_mom for the first month; drop if missing
        q4_top = q4_top[~q4_top['pct_change_mom'].isna()]

        if q4_top.empty:
            st.info("Insufficient MoM change data to determine Q4 top months.")
        else:
            q4_top['month_name'] = q4_top['date'].dt.strftime('%b')
            q4_display = q4_top[['year', 'month_name', 'pct_change_mom']].copy()
            q4_display = q4_display.sort_values('year')
            q4_display['pct_change_mom'] = q4_display['pct_change_mom'].round(2)

            # Build HTML table
            table_rows = []
            for _, r in q4_display.iterrows():
                table_rows.append(f"<tr><td style='padding:8px;border-bottom:1px solid #eee;'>{int(r['year'])}</td><td style='padding:8px;border-bottom:1px solid #eee;'><strong>{r['month_name']}</strong></td><td style='padding:8px;border-bottom:1px solid #eee;color:#065f46;font-weight:600;'>{r['pct_change_mom']}%</td></tr>")

            summary_html = (
                "<style>.q4-table{width:100%;border-collapse:collapse;font-family:Arial,sans-serif}.q4-table thead th{background:#0ea5a4;color:white;padding:8px;text-align:left}.q4-table tbody td{padding:8px;color:#063047}.q4-note{margin-top:8px;font-size:0.95rem;color:#334155}</style>"
            )

            table_html = (
                summary_html
                + "<table class='q4-table'><thead><tr><th>Year</th><th>Top Q4 Month</th><th>MoM Growth (%)</th></tr></thead><tbody>"
                + "".join(table_rows)
                + "</tbody></table>"
            )

            # Frequency summary: which month appears most often as the top month in Q4
            freq = q4_display['month_name'].value_counts()
            top_freq = freq.idxmax()
            top_freq_count = int(freq.max())
            total_years = int(freq.sum())
            freq_line = f"<div class='q4-note'><strong>Summary:</strong> <span style='color:#0f766e;font-weight:700;'>{top_freq}</span> was the top Q4 month in <strong>{top_freq_count}</strong> out of <strong>{total_years}</strong> years ({(top_freq_count/total_years*100):.1f}%).</div>"

            st.markdown(table_html + freq_line, unsafe_allow_html=True)

def create_roll_up_charts(df):
    """Performs and visualizes Roll-up analyses."""
    st.header("Roll-up Analysis")

    # Roll-up 1: Quarter-over-quarter and Year-over-year growth rates
    st.subheader("Quarter-over-Quarter (Q-o-Q) Employment Analysis")

    # Quarterly aggregation
    df_quarterly = df.set_index('date').resample('QS').mean()
    df_quarterly['quarter'] = df_quarterly.index.quarter
    df_quarterly['year'] = df_quarterly.index.year
    df_quarterly['qoq_growth'] = df_quarterly['total_nonfarm'].pct_change() * 100

    # Checkbox interface for selecting quarters
    quarter_options = [1, 2, 3, 4]
    quarter_labels = [f"Q{q}" for q in quarter_options]
    selected_quarters = st.multiselect(
        "Select Quarters to Display:",
        options=quarter_options,
        default=quarter_options,
        format_func=lambda x: f"Q{x}"
    )

    # Year slider for interval selection
    min_year = int(df_quarterly['year'].min())
    max_year = int(df_quarterly['year'].max())
    year_range = st.slider(
        "Select year range for quarterly analysis:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )

    # Filter for selected quarters and year interval
    df_quarter_filtered = df_quarterly[
        (df_quarterly['quarter'].isin(selected_quarters)) &
        (df_quarterly['year'] >= year_range[0]) &
        (df_quarterly['year'] <= year_range[1])
    ].reset_index()

    # Line chart: one line per selected quarter
    fig_qoq = px.line(
        df_quarter_filtered,
        x='year',
        y='qoq_growth',
        color='quarter',
        markers=True,
        title=f"Quarter-over-Quarter Employment Growth Rate by Quarter",
        labels={'year': 'Year', 'qoq_growth': 'QoQ Growth (%)', 'quarter': 'Quarter'},
        color_discrete_map={1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}
    )
    st.plotly_chart(fig_qoq)

    # Annual Analysis
    st.subheader("Annual Analysis")
    # Yearly aggregation
    df_yearly = df.set_index('date').resample('A').mean()
    df_yearly['year'] = df_yearly.index.year
    df_yearly['yoy_growth'] = df_yearly['total_nonfarm'].pct_change() * 100
    min_year_annual = int(df_yearly['year'].min())
    max_year_annual = int(df_yearly['year'].max())
    year_range_annual = st.slider(
        "Select year range for annual analysis:",
        min_value=min_year_annual,
        max_value=max_year_annual,
        value=(min_year_annual, max_year_annual),
        step=1
    )
    df_yearly_interval = df_yearly[(df_yearly['year'] >= year_range_annual[0]) & (df_yearly['year'] <= year_range_annual[1])].reset_index()
    fig_yoy = px.line(
        df_yearly_interval,
        x='year',
        y='yoy_growth',
        title="Year-over-Year Employment Growth Rate",
        labels={'year': 'Year', 'yoy_growth': 'YoY Growth (%)'}
    )
    st.plotly_chart(fig_yoy)
    
    # Roll-up 2: Compare average employment in 2010s vs. 2000s
    st.subheader("Average Employment in the 2000s vs. the 2010s")
    df['year'] = df['date'].dt.year
    decade_2000s = df[(df['year'] >= 2000) & (df['year'] <= 2009)]
    decade_2010s = df[(df['year'] >= 2010) & (df['year'] <= 2019)]
    
    avg_2000s = decade_2000s['total_nonfarm'].mean()
    avg_2010s = decade_2010s['total_nonfarm'].mean()
    
    comparison_df = pd.DataFrame({
        'Decade': ['2000s', '2010s'],
        'Average Employment': [avg_2000s, avg_2010s]
    })
    
    fig_decades = px.bar(comparison_df, x='Decade', y='Average Employment', 
                         title="Average Employment: 2000s vs. 2010s",
                         labels={'Average Employment': 'Average Employment (in thousands)'})
    st.plotly_chart(fig_decades)

    # Roll-up 3: Net Jobs Created per Decade (1940s - 2010s, include 2020 if present)
    st.subheader("Net Jobs Created per Decade (1940-2020)")

    # Ensure date/year columns and sort by date
    df = df.copy()
    df = df.sort_values('date')
    df['year'] = df['date'].dt.year

    # Build decade ranges: 1940-1949, 1950-1959, ..., 2010-2019. Include 2020 as its own bucket if present.
    decades = []
    for start in range(1940, 2011, 10):
        end = start + 9
        decades.append((start, end))
    # If dataset contains year 2020, include it as a single-year bucket (user asked up to 2020)
    include_2020 = df['year'].isin([2020]).any()
    if include_2020:
        decades.append((2020, 2020))

    decade_labels = []
    net_jobs = []
    for start, end in decades:
        mask = (df['year'] >= start) & (df['year'] <= end)
        seg = df[mask].sort_values('date')
        label = f"{start}s" if start != 2020 else '2020'
        if seg.empty:
            # No data for this decade; report 0 net change
            decade_labels.append(label)
            net_jobs.append(0.0)
        else:
            # Net jobs created = last observed total_nonfarm - first observed total_nonfarm
            net_change = seg['total_nonfarm'].iloc[-1] - seg['total_nonfarm'].iloc[0]
            decade_labels.append(label)
            net_jobs.append(net_change)

    dec_df = pd.DataFrame({'Decade': decade_labels, 'Net Jobs Created': net_jobs})

    # Use a qualitative color sequence with one color per bar
    colors = px.colors.qualitative.Plotly
    # Extend or cycle colors to match number of decades
    color_seq = [colors[i % len(colors)] for i in range(len(dec_df))]

    fig_decade_net = px.bar(
        dec_df,
        x='Decade',
        y='Net Jobs Created',
        title='Net Jobs Created by Decade (1940–2020)',
        labels={'Net Jobs Created': 'Net Jobs Created (thousands)'},
        color='Decade',
        color_discrete_sequence=color_seq
    )
    fig_decade_net.update_traces(marker_line_width=0.5)
    fig_decade_net.update_layout(showlegend=False)
    st.plotly_chart(fig_decade_net, use_container_width=True)

def create_drill_down_charts(df):
    """Performs and visualizes Drill-down analyses."""
    st.header("Drill-down Analysis")
    
    # Drill-down 1: Year with highest annual employment gain
    st.subheader("Breakdown of Highest Annual Employment Gain")
    df_annual = df.groupby(df['date'].dt.year)['total_nonfarm'].sum().reset_index()
    df_annual['annual_gain'] = df_annual['total_nonfarm'].diff()
    df_annual.columns = ['year', 'total_employment', 'annual_gain']
    highest_gain_year = df_annual.loc[df_annual['annual_gain'].idxmax()]['year']

    st.write(f"The year with the highest annual employment gain was **{int(highest_gain_year)}**.")

    # Drill-down into that year's monthly contributions
    highest_gain_df = df[df['date'].dt.year == highest_gain_year].copy()
    highest_gain_df['month'] = highest_gain_df['date'].dt.strftime('%b')
    highest_gain_df['quarter'] = highest_gain_df['date'].dt.quarter

    # Chart above, facts below
    view_option = st.radio("View breakdown by:", options=["Month", "Quarter"], index=0)
    if view_option == "Month":
        fig_drill = px.line(
            highest_gain_df,
            x='month',
            y='total_nonfarm',
            markers=True,
            title=f"Monthly Employment Contributions in {int(highest_gain_year)}",
            labels={'total_nonfarm': 'Total Employment (in thousands)', 'month': 'Month'}
        )
    else:
        quarterly_df = highest_gain_df.groupby('quarter')['total_nonfarm'].sum().reset_index()
        fig_drill = px.line(
            quarterly_df,
            x='quarter',
            y='total_nonfarm',
            markers=True,
            title=f"Quarterly Employment Contributions in {int(highest_gain_year)}",
            labels={'total_nonfarm': 'Total Employment (in thousands)', 'quarter': 'Quarter'}
        )
    st.plotly_chart(fig_drill, use_container_width=True)

    # Facts section below chart, with CSS styling
    if int(highest_gain_year) == 2022:
        st.markdown("""
<div class="facts-section">
<strong>Facts about the USA in 2022 (Highest Employment Gain Year):</strong>
<ul>
<li><strong>Major Job Providing Sectors:</strong>
    <ul>
        <li>Healthcare & Social Assistance</li>
        <li>Professional & Business Services</li>
        <li>Leisure & Hospitality</li>
        <li>Retail Trade</li>
        <li>Construction</li>
    </ul>
</li>
<li><strong>Leading Companies Hiring in 2022:</strong>
    <ul>
        <li>Amazon</li>
        <li>Walmart</li>
        <li>CVS Health</li>
        <li>McDonald's</li>
        <li>Microsoft, Google, Apple</li>
    </ul>
</li>
<li><strong>Economic Context:</strong>
    <ul>
        <li>Strong labor market recovery post-pandemic</li>
        <li>Wage growth, especially in lower-wage sectors</li>
        <li>Remote and hybrid work models became mainstream</li>
    </ul>
</li>
</ul>
<em>Sources: U.S. Bureau of Labor Statistics, Reuters, CNBC, Bloomberg, company press releases.</em>
</div>
<style>
.facts-section {
    border: 2px solid #4F8BF9;
    border-radius: 12px;
    background: #f9fbff;
    padding: 24px 20px 16px 20px;
    margin-top: 32px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(79,139,249,0.08);
    font-size: 1.08rem;
    color: #222;
}
.facts-section strong {
    color: #4F8BF9;
    font-size: 1.15rem;
}
.facts-section ul {
    margin-left: 0.5em;
    margin-bottom: 0.5em;
}
.facts-section li {
    margin-bottom: 0.25em;
}
.facts-section em {
    color: #888;
    font-size: 0.98rem;
}
</style>
        """, unsafe_allow_html=True)
    
    # Drill-down 2: Sharpest monthly drop
    st.subheader("Sharpest Monthly Employment Drop")
    df['mom_drop'] = df['total_nonfarm'].diff()
    sharpest_drop_month = df.loc[df['mom_drop'].idxmin()]
    
    st.write(f"The sharpest drop in employment occurred in **{sharpest_drop_month['date'].strftime('%B %Y')}**.")
    st.write(f"The total payroll employment decreased by approximately **{sharpest_drop_month['mom_drop']:.2f} thousand** that month.")
    
    # Attempt to provide a weekly breakdown for that month if higher-frequency data exists
    target_year = int(sharpest_drop_month['date'].year)
    target_month = int(sharpest_drop_month['date'].month)

    # Subset rows that fall in the same month/year
    df_month_rows = df[(df['date'].dt.year == target_year) & (df['date'].dt.month == target_month)].copy()

    if df_month_rows.shape[0] <= 1:
        # Most datasets here are monthly; explicitly state that weekly breakdown isn't available
        st.info("Weekly breakdown not available: dataset appears to be monthly (one row for the month).")
    else:
        # We have higher-frequency rows within the month (daily/weekly). Aggregate to ISO-week and compute week-over-week drops
        df_month_rows = df_month_rows.set_index('date').sort_index()
        # Sum (or mean) total_nonfarm per week (weeks ending on Mon to keep consistency)
        weekly = df_month_rows['total_nonfarm'].resample('W-MON').sum().reset_index()
        if weekly.shape[0] <= 1:
            st.info("Insufficient weekly data points within that month to compute week-over-week changes.")
        else:
            weekly['w_change'] = weekly['total_nonfarm'].diff()
            weekly['w_change_pct'] = weekly['total_nonfarm'].pct_change() * 100

            # Restrict to weeks that overlap the target month (some weekly bins may spill into adjacent months)
            weekly_in_month = weekly[weekly['date'].dt.to_period('M') == pd.Period(f"{target_year}-{target_month:02d}")] 
            if weekly_in_month.empty:
                # As a fallback, pick weeks where the week midpoint falls in the month
                weekly['midpoint'] = weekly['date'] - pd.to_timedelta(3, unit='d')
                weekly_in_month = weekly[weekly['midpoint'].dt.to_period('M') == pd.Period(f"{target_year}-{target_month:02d}")]

            if weekly_in_month.empty:
                st.info("Could not isolate weekly bins for that month. Weekly breakdown unavailable.")
            else:
                # Find the weeks with the largest negative absolute drop (most affected)
                most_affected = weekly_in_month.nsmallest(3, 'w_change').copy()
                # Build human-friendly labels
                rows = []
                for _, r in most_affected.iterrows():
                    week_label = r['date'].strftime('%Y-%m-%d')
                    drop_val = r['w_change']
                    drop_pct = r['w_change_pct']
                    rows.append((week_label, drop_val, drop_pct))

                # Render as a small HTML snippet (bold statement + table)
                rows_html = ""
                for wl, dv, dp in rows:
                    rows_html += f"<tr><td style='padding:6px;border-bottom:1px solid #eee;'>{wl}</td><td style='padding:6px;border-bottom:1px solid #eee;color:#b91c1c;font-weight:700;'>{dv:.1f}</td><td style='padding:6px;border-bottom:1px solid #eee;color:#7c2d12;'>{dp:.1f}%</td></tr>"

                html = (
                    "<div style='margin-top:12px;'><strong>Weeks most affected within that month (top declines):</strong>" 
                    "<table style='width:100%;border-collapse:collapse;margin-top:8px;font-family:Arial,sans-serif;'>"
                    "<thead><tr><th style='background:#fecaca;padding:6px;text-align:left;color:#7f1d1d;'>Week Ending</th>"
                    "<th style='background:#fecaca;padding:6px;text-align:left;color:#7f1d1d;'>Absolute Drop</th>"
                    "<th style='background:#fecaca;padding:6px;text-align:left;color:#7f1d1d;'>Pct Change</th></tr></thead><tbody>"
                    + rows_html + "</tbody></table></div>"
                )
                st.markdown(html, unsafe_allow_html=True)
    
                # --- New: For every year, compute top contributing months and render a focused table for the highest-gain year ---
                st.subheader("Top Monthly Contributors by Year")
                # Prepare monthly sums per year
                df_monthly_all = df.copy()
                df_monthly_all['year'] = df_monthly_all['date'].dt.year
                df_monthly_all['month_num'] = df_monthly_all['date'].dt.month
                df_monthly_all['month'] = df_monthly_all['date'].dt.strftime('%b')

                monthly_by_year = df_monthly_all.groupby(['year', 'month_num', 'month'])['total_nonfarm'].sum().reset_index()

                # Compute year-over-year contributions: for each year, contribution = month_this_year - same_month_prev_year
                years = sorted(monthly_by_year['year'].unique())
                contributors = []
                for y in years:
                    this_year = monthly_by_year[monthly_by_year['year'] == y].set_index('month_num')
                    prev_year = monthly_by_year[monthly_by_year['year'] == (y - 1)].set_index('month_num')
                    merged = this_year.join(prev_year[['total_nonfarm']], how='left', rsuffix='_prev')
                    merged = merged.reset_index()
                    merged['total_nonfarm_prev'] = merged['total_nonfarm_prev'].fillna(0)
                    merged['contribution'] = merged['total_nonfarm'] - merged['total_nonfarm_prev']
                    # Top 3 months for this year
                    top3 = merged.sort_values('contribution', ascending=False).head(3)
                    top_list = ", ".join([f"{r['month']} ({r['contribution']:.1f})" for _, r in top3.iterrows()])
                    total_gain = merged['contribution'].sum()
                    contributors.append({'year': y, 'total_gain': total_gain, 'top_months': top_list})

                contrib_df = pd.DataFrame(contributors)
                contrib_df['total_gain'] = contrib_df['total_gain'].round(1)

                # Build HTML table listing year, total annual gain, and top 3 months (with contributions)
                rows = []
                for _, r in contrib_df.sort_values('year', ascending=False).iterrows():
                    rows.append(f"<tr><td style='padding:8px;border-bottom:1px solid #eee;'>{int(r['year'])}</td><td style='padding:8px;border-bottom:1px solid #eee;text-align:right;color:#065f46;font-weight:700;'>{r['total_gain']}</td><td style='padding:8px;border-bottom:1px solid #eee;'>{r['top_months']}</td></tr>")

                html = ("<style>.year-contrib{width:100%;border-collapse:collapse;font-family:Arial,sans-serif}.year-contrib thead th{background:#7c3aed;color:white;padding:8px;text-align:left}</style>"
                        + "<table class='year-contrib'><thead><tr><th>Year</th><th style='text-align:right;'>Annual Gain</th><th>Top 3 Contributing Months (contribution)</th></tr></thead><tbody>"
                        + "".join(rows)
                        + "</tbody></table>")
                st.markdown(html, unsafe_allow_html=True)

                # Focused: detailed month-by-month table for the highest-gain year
                try:
                    highest_year = int(highest_gain_year)
                    st.markdown(f"### Detailed Monthly Contributions for {highest_year}")
                    detailed_this = monthly_by_year[monthly_by_year['year'] == highest_year].set_index('month_num')
                    detailed_prev = monthly_by_year[monthly_by_year['year'] == (highest_year - 1)].set_index('month_num')
                    det = detailed_this.join(detailed_prev[['total_nonfarm']], how='left', rsuffix='_prev').reset_index()
                    det['total_nonfarm_prev'] = det['total_nonfarm_prev'].fillna(0)
                    det['contribution'] = det['total_nonfarm'] - det['total_nonfarm_prev']
                    det = det.sort_values('month_num')

                    det['total_nonfarm'] = det['total_nonfarm'].apply(lambda x: f"{x:,.1f}")
                    det['total_nonfarm_prev'] = det['total_nonfarm_prev'].apply(lambda x: f"{x:,.1f}")
                    det['contribution'] = det['contribution'].apply(lambda x: f"{x:,.1f}")

                    # Build HTML
                    rows = []
                    for _, r in det.iterrows():
                        month = pd.to_datetime(int(r['month_num']), format='%m').strftime('%b')
                        rows.append(f"<tr><td style='padding:8px;border-bottom:1px solid #eee;'>{month}</td><td style='padding:8px;border-bottom:1px solid #eee;text-align:right;'>{r['total_nonfarm']}</td><td style='padding:8px;border-bottom:1px solid #eee;text-align:right;'>{r['total_nonfarm_prev']}</td><td style='padding:8px;border-bottom:1px solid #eee;text-align:right;color:#065f46;font-weight:700;'>{r['contribution']}</td></tr>")

                    html_det = ("<style>.det-table{width:100%;border-collapse:collapse;font-family:Arial,sans-serif}.det-table thead th{background:#0ea5a4;color:white;padding:8px;text-align:left}</style>"
                                + "<table class='det-table'><thead><tr><th>Month</th><th style='text-align:right;'>Current</th><th style='text-align:right;'>Same Month Prev Year</th><th style='text-align:right;'>Contribution</th></tr></thead><tbody>"
                                + "".join(rows)
                                + "</tbody></table>")
                    st.markdown(html_det, unsafe_allow_html=True)
                except Exception:
                    st.info("Unable to build detailed contributors table for the highest-gain year.")


# --- 4. Main App Structure ---
def main():
    add_custom_css()
    st.title("U.S. Non-Farm Payrolls OLAP Analysis")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    menu_selection = st.sidebar.radio(
        "Select an analysis type:",
        ["Slicing", "Dicing", "Roll-up", "Drill-Down"]
    )

    data = load_data()

    if data is not None:
        if menu_selection == "Slicing":
            create_slicing_charts(data.copy())
        elif menu_selection == "Dicing":
            create_dicing_charts(data.copy())
        elif menu_selection == "Roll-up":
            create_roll_up_charts(data.copy())
        elif menu_selection == "Drill-Down":
            create_drill_down_charts(data.copy())

if __name__ == "__main__":
    main()
