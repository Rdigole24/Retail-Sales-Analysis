import streamlit as st
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import sys
import os

# Add the current directory to the path to import forecasting module
# This assumes forecasting.py is in the same directory as this dashboard file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from forecasting import get_sales_forecasts, prepare_time_series_data
except ImportError:
    st.error("Error: 'forecasting.py' not found. Please ensure it's in the same directory.")
    st.stop()

# Database connection details
DB_CONFIG = {
    'host': 'localhost',
    'database': 'retail_sales',
    'user': 'root',
    'password': 'root'
}

@st.cache_data
def get_data_from_db(query):
    """Execute SQL query and return DataFrame"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        df = pd.read_sql(query, connection)
        connection.close()
        return df
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return pd.DataFrame()

def execute_custom_query(query):
    """Execute custom SQL query and return results"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        start_time = time.time()
        cursor.execute(query)
        execution_time = time.time() - start_time
        
        if query.strip().upper().startswith('SELECT') or query.strip().upper().startswith('CALL'):
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            connection.close()
            return df, execution_time, None
        else:
            connection.commit()
            affected_rows = cursor.rowcount
            connection.close()
            return None, execution_time, f"Query executed successfully. {affected_rows} rows affected."
    except Exception as e:
        return None, 0, f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="🚀 Unified Retail Sales Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚀 Retail Sales Insights Dashboard")
    st.markdown("### *SQL Analytics with Predictive Capabilities*")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Executive Overview", 
        "💰 Sales Analytics", 
        "📦 Product Intelligence", 
        "👤 Customer Insights", 
        "🌍 Regional Performance",
        "🔍 SQL Query Lab",
        "📈 Advanced Analytics",
        "🔮 Predictive Forecasting",
        "⚡ Performance Dashboard",
        "📋 Data Export Center"
    ])

    if page == "📊 Executive Overview":
        show_executive_overview()
    elif page == "💰 Sales Analytics":
        show_sales_analytics()
    elif page == "📦 Product Intelligence":
        show_product_intelligence()
    elif page == "👤 Customer Insights":
        show_customer_insights()
    elif page == "🌍 Regional Performance":
        show_regional_performance()
    elif page == "🔍 SQL Query Lab":
        show_sql_query_lab()
    elif page == "📈 Advanced Analytics":
        show_advanced_analytics()
    elif page == "🔮 Predictive Forecasting":
        show_predictive_forecasting()
    elif page == "⚡ Performance Dashboard":
        show_performance_dashboard()
    elif page == "📋 Data Export Center":
        show_data_export_center()

def show_executive_overview():
    st.header("📈 Executive Overview")
    
    # Advanced date range filter with presets
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        date_preset = st.selectbox("Quick Date Range:", [
            "All Time", "Last 30 Days", "Last 90 Days", "Last 6 Months", 
            "Last Year", "This Year", "Custom Range"
        ])
    
    # Calculate date range based on preset
    # Assuming the latest data in your DB is around 2017-12-31
    max_order_date_query = "SELECT MAX(order_date) FROM orders;"
    max_date_df = get_data_from_db(max_order_date_query)
    end_date_val = max_date_df.iloc[0, 0] if not max_date_df.empty and max_date_df.iloc[0, 0] else datetime(2017, 12, 31).date()
    
    if date_preset == "Last 30 Days":
        start_date_val = end_date_val - timedelta(days=30)
    elif date_preset == "Last 90 Days":
        start_date_val = end_date_val - timedelta(days=90)
    elif date_preset == "Last 6 Months":
        start_date_val = end_date_val - timedelta(days=180)
    elif date_preset == "Last Year":
        start_date_val = end_date_val - timedelta(days=365)
    elif date_preset == "This Year":
        start_date_val = datetime(end_date_val.year, 1, 1).date()
    elif date_preset == "Custom Range":
        with col2:
            start_date_val = st.date_input("Start Date", value=datetime(2014, 1, 1).date())
        with col3:
            end_date_val = st.date_input("End Date", value=end_date_val)
    else:  # All Time
        min_order_date_query = "SELECT MIN(order_date) FROM orders;"
        min_date_df = get_data_from_db(min_order_date_query)
        start_date_val = min_date_df.iloc[0, 0] if not min_date_df.empty and min_date_df.iloc[0, 0] else datetime(2014, 1, 1).date()
    
    # Key Performance Indicators
    st.subheader("🎯 Key Performance Indicators")
    
    # Enhanced metrics with comparisons
    metrics_query = f"""
    SELECT 
        SUM(s.sales) as total_sales,
        SUM(s.profit) as total_profit,
        COUNT(DISTINCT s.order_id) as total_orders,
        COUNT(DISTINCT o.customer_id) as unique_customers,
        AVG(s.sales) as avg_order_value,
        SUM(s.profit) / SUM(s.sales) * 100 as profit_margin
    FROM sales s 
    JOIN orders o ON s.order_id = o.order_id 
    WHERE o.order_date BETWEEN '{start_date_val}' AND '{end_date_val}'
    """
    
    metrics_df = get_data_from_db(metrics_query)
    
    if not metrics_df.empty:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        total_sales = metrics_df['total_sales'].iloc[0] or 0
        total_profit = metrics_df['total_profit'].iloc[0] or 0
        total_orders = metrics_df['total_orders'].iloc[0] or 0
        unique_customers = metrics_df['unique_customers'].iloc[0] or 0
        avg_order_value = metrics_df['avg_order_value'].iloc[0] or 0
        profit_margin = metrics_df['profit_margin'].iloc[0] or 0
        
        col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
        col2.metric("📈 Total Profit", f"${total_profit:,.0f}")
        col3.metric("🛒 Total Orders", f"{total_orders:,}")
        col4.metric("👥 Customers", f"{unique_customers:,}")
        col5.metric("💳 Avg Order Value", f"${avg_order_value:.2f}")
        col6.metric("📊 Profit Margin", f"{profit_margin:.1f}%")

    st.markdown("---")

    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Sales Trend Analysis")
        # Monthly sales trend using view
        # Ensure the view 'monthly_sales_profit_view' exists in your database
        monthly_trend_df = get_data_from_db("SELECT * FROM monthly_sales_profit_view ORDER BY sales_month")
        
        if not monthly_trend_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_trend_df['sales_month'], 
                y=monthly_trend_df['monthly_sales'],
                mode='lines+markers',
                name='Sales',
                line=dict(color='#1f77b4', width=3),
                fill='tozeroy' # Changed from 'tonexty' for better single-trace fill
            ))
            fig.add_trace(go.Scatter(
                x=monthly_trend_df['sales_month'], 
                y=monthly_trend_df['monthly_profit'],
                mode='lines+markers',
                name='Profit',
                line=dict(color='#2ca02c', width=3),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Monthly Sales and Profit Trend',
                xaxis_title='Month',
                yaxis=dict(title='Sales ($)', side='left'),
                yaxis2=dict(title='Profit ($)', side='right', overlaying='y'),
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monthly sales trend data available. Please ensure 'monthly_sales_profit_view' exists and has data.")
    
    with col2:
        st.subheader("🏆 Top Performance Categories")
        # Category performance using view
        # Ensure the view 'sales_by_category_view' exists in your database
        category_df = get_data_from_db("SELECT * FROM sales_by_category_view ORDER BY total_sales DESC LIMIT 10")
        
        if not category_df.empty:
            fig = px.treemap(
                category_df, 
                path=['category'], 
                values='total_sales',
                title='Sales by Category',
                color='total_profit',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category sales data available. Please ensure 'sales_by_category_view' exists and has data.")

def show_sales_analytics():
    st.header("💰 Sales Analytics")
    
    # Sales by Ship Mode
    st.subheader("🚚 Sales by Ship Mode")
    ship_mode_query = """
    SELECT 
        o.ship_mode,
        SUM(s.sales) AS total_sales,
        SUM(s.profit) AS total_profit
    FROM 
        sales s
    JOIN 
        orders o ON s.order_id = o.order_id
    GROUP BY 
        o.ship_mode
    ORDER BY 
        total_sales DESC
    """
    ship_mode_df = get_data_from_db(ship_mode_query)
    
    if not ship_mode_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(ship_mode_df, x='ship_mode', y='total_sales',
                        title='Sales by Ship Mode',
                        labels={'ship_mode': 'Ship Mode', 'total_sales': 'Total Sales ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(ship_mode_df, values='total_sales', names='ship_mode',
                        title='Sales Distribution by Ship Mode')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No ship mode data available.")

    # Sales and Profit by Segment
    st.subheader("👥 Sales by Customer Segment")
    segment_query = """
    SELECT 
        o.segment,
        SUM(s.sales) AS total_sales,
        SUM(s.profit) AS total_profit
    FROM 
        sales s
    JOIN 
        orders o ON s.order_id = o.order_id
    GROUP BY 
        o.segment
    ORDER BY 
        total_sales DESC
    """
    segment_df = get_data_from_db(segment_query)
    
    if not segment_df.empty:
        fig = px.bar(segment_df, x='segment', y=['total_sales', 'total_profit'],
                    title='Sales and Profit by Customer Segment',
                    labels={'value': 'Amount ($)', 'segment': 'Customer Segment'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No customer segment data available.")

    # Advanced filtering (from advanced_dashboard)
    st.subheader("🔍 Advanced Sales Filtering")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Fetch distinct regions dynamically
        regions_df = get_data_from_db("SELECT DISTINCT region FROM orders")
        regions = ["All"] + (regions_df['region'].tolist() if not regions_df.empty else [])
        selected_region = st.selectbox("Select Region:", regions)
    with col2:
        # Fetch distinct categories dynamically
        categories_df = get_data_from_db("SELECT DISTINCT category FROM products")
        categories = ["All"] + (categories_df['category'].tolist() if not categories_df.empty else [])
        selected_category = st.selectbox("Select Category:", categories)
    with col3:
        min_sales = st.number_input("Minimum Sales Amount:", min_value=0.0, value=0.0)
    
    # Build dynamic query based on filters
    where_conditions = []
    if selected_region != "All":
        where_conditions.append(f"o.region = '{selected_region}'")
    if selected_category != "All":
        where_conditions.append(f"p.category = '{selected_category}'")
    if min_sales > 0:
        where_conditions.append(f"s.sales >= {min_sales}")
    
    where_clause = " AND " + " AND ".join(where_conditions) if where_conditions else ""
    
    filtered_query = f"""
    SELECT 
        o.region,
        p.category,
        SUM(s.sales) as total_sales,
        SUM(s.profit) as total_profit,
        COUNT(*) as transaction_count
    FROM sales s
    JOIN orders o ON s.order_id = o.order_id
    JOIN products p ON s.product_id = p.product_id
    WHERE 1=1 {where_clause}
    GROUP BY o.region, p.category
    ORDER BY total_sales DESC
    """
    
    filtered_df = get_data_from_db(filtered_query)
    if not filtered_df.empty:
        st.subheader("📈 Filtered Results")
        st.dataframe(filtered_df, use_container_width=True)
        
        if len(filtered_df) > 0:
            fig = px.treemap(
                filtered_df, 
                path=['region', 'category'], 
                values='total_sales',
                title='Sales Treemap by Region and Category'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data found for the selected filters.")

def show_product_intelligence():
    st.header("📦 Product Intelligence")
    
    # Sales by Category (from dashboard.py, but using view if available)
    st.subheader("📊 Sales by Product Category")
    # Prioritize view from advanced_dashboard if it exists
    category_df = get_data_from_db("SELECT * FROM sales_by_category_view ORDER BY total_sales DESC")
    
    if not category_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(category_df, x='category', y='total_sales',
                        title='Sales by Category',
                        labels={'category': 'Category', 'total_sales': 'Total Sales ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(category_df, x='total_sales', y='total_profit', 
                           size='total_quantity' if 'total_quantity' in category_df.columns else None, 
                           hover_name='category',
                           title='Sales vs Profit by Category',
                           labels={'total_sales': 'Total Sales ($)', 'total_profit': 'Total Profit ($)'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No product category data available. Ensure 'sales_by_category_view' exists or data is present.")

    # Top Products (Using Stored Procedure from advanced_dashboard)
    st.subheader("🏆 Top Products (Using Stored Procedure)")
    
    n_products = st.slider("Number of top products:", min_value=5, max_value=50, value=10)
    
    if st.button("🔍 Get Top Products"):
        # Ensure 'GetTopNProductsBySales' stored procedure exists in your database
        top_products_df = get_data_from_db(f"CALL GetTopNProductsBySales({n_products})")
        if not top_products_df.empty:
            fig = px.bar(
                top_products_df, 
                x='total_sales', 
                y='product_name',
                orientation='h',
                title=f'Top {n_products} Products by Sales'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top_products_df, use_container_width=True)
        else:
            st.info(f"No top {n_products} products found. Ensure 'GetTopNProductsBySales' stored procedure exists and returns data.")

    # Top Sub-Categories (from dashboard.py)
    st.subheader("📈 Top 10 Sub-Categories by Sales")
    subcategory_query = """
    SELECT 
        p.sub_category,
        SUM(s.sales) AS total_sales,
        SUM(s.profit) AS total_profit
    FROM 
        sales s
    JOIN 
        products p ON s.product_id = p.product_id
    GROUP BY 
        p.sub_category
    ORDER BY 
        total_sales DESC
    LIMIT 10
    """
    subcategory_df = get_data_from_db(subcategory_query)
    
    if not subcategory_df.empty:
        fig = px.bar(subcategory_df, x='total_sales', y='sub_category',
                    orientation='h',
                    title='Top 10 Sub-Categories by Sales',
                    labels={'total_sales': 'Total Sales ($)', 'sub_category': 'Sub-Category'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sub-category data available.")

def show_customer_insights():
    st.header("👤 Customer Insights")
    
    # Top Customers (from dashboard.py)
    st.subheader("🌟 Top 10 Customers by Sales")
    top_customers_query = """
    SELECT 
        o.customer_name,
        SUM(s.sales) AS total_sales_by_customer,
        SUM(s.profit) AS total_profit_by_customer,
        COUNT(DISTINCT s.order_id) AS order_count
    FROM 
        sales s
    JOIN 
        orders o ON s.order_id = o.order_id
    GROUP BY 
        o.customer_name
    ORDER BY 
        total_sales_by_customer DESC
    LIMIT 10
    """
    top_customers_df = get_data_from_db(top_customers_query)
    
    if not top_customers_df.empty:
        fig = px.bar(top_customers_df, x='total_sales_by_customer', y='customer_name',
                    orientation='h',
                    title='Top 10 Customers by Sales',
                    labels={'total_sales_by_customer': 'Total Sales ($)', 'customer_name': 'Customer Name'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.subheader("📋 Customer Details")
        st.dataframe(top_customers_df, use_container_width=True)
    else:
        st.info("No top customer data available.")

    # Customer segmentation (from advanced_dashboard)
    st.subheader("🎯 Customer Segmentation")
    
    segmentation_query = """
    WITH CustomerMetrics AS (
        SELECT 
            o.customer_id,
            o.customer_name,
            COUNT(DISTINCT s.order_id) as order_frequency,
            SUM(s.sales) as total_sales,
            AVG(s.sales) as avg_order_value,
            DATEDIFF(CURDATE(), MAX(o.order_date)) as days_since_last_order
        FROM sales s
        JOIN orders o ON s.order_id = o.order_id
        GROUP BY o.customer_id, o.customer_name
    )
    SELECT 
        customer_name,
        order_frequency,
        total_sales,
        avg_order_value,
        days_since_last_order,
        CASE 
            WHEN order_frequency >= 10 AND total_sales >= 1000 THEN 'VIP'
            WHEN order_frequency >= 5 AND total_sales >= 500 THEN 'Loyal'
            WHEN order_frequency >= 2 AND total_sales >= 200 THEN 'Regular'
            ELSE 'New'
        END as customer_segment
    FROM CustomerMetrics
    ORDER BY total_sales DESC
    LIMIT 100;
    """
    
    segmentation_df = get_data_from_db(segmentation_query)
    if not segmentation_df.empty:
        # Customer segment distribution
        segment_counts = segmentation_df['customer_segment'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                        title='Customer Segment Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                segmentation_df, 
                x='order_frequency', 
                y='total_sales',
                color='customer_segment',
                hover_name='customer_name',
                title='Customer Segmentation Analysis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(segmentation_df, use_container_width=True)
    else:
        st.info("No customer segmentation data available.")

    # Customer Lifetime Value Analysis (from advanced_dashboard)
    st.subheader("💎 Customer Lifetime Value Analysis")
    
    clv_query = """
    WITH CustomerSales AS (
        SELECT
            o.customer_id,
            o.customer_name,
            SUM(s.sales) AS total_sales,
            COUNT(DISTINCT s.order_id) AS total_orders,
            MIN(o.order_date) AS first_order,
            MAX(o.order_date) AS last_order
        FROM sales s
        JOIN orders o ON s.order_id = o.order_id
        GROUP BY o.customer_id, o.customer_name
    ),
    RankedCustomers AS (
        SELECT
            customer_id,
            customer_name,
            total_sales,
            total_orders,
            first_order,
            last_order,
            DATEDIFF(last_order, first_order) AS customer_lifespan_days,
            ROW_NUMBER() OVER (ORDER BY total_sales DESC) as sales_rank
        FROM CustomerSales
    )
    SELECT
        customer_name,
        total_sales,
        total_orders,
        sales_rank,
        customer_lifespan_days,
        (total_sales / total_orders) AS average_order_value,
        CASE 
            WHEN customer_lifespan_days > 0 
            THEN (total_sales / customer_lifespan_days) * 365 
            ELSE total_sales 
        END AS estimated_annual_value
    FROM RankedCustomers
    ORDER BY estimated_annual_value DESC
    LIMIT 20;
    """
    
    clv_df = get_data_from_db(clv_query)
    if not clv_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                clv_df, 
                x='average_order_value', 
                y='estimated_annual_value',
                size='total_orders',
                hover_name='customer_name',
                title='Customer Value Analysis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(clv_df[['customer_name', 'total_sales', 'estimated_annual_value']], use_container_width=True)
    else:
        st.info("No customer lifetime value data available.")

def show_regional_performance():
    st.header("🌍 Regional Performance")
    
    # Sales by Region (from dashboard.py)
    st.subheader("🗺️ Sales and Profit by Region")
    region_query = """
    SELECT 
        o.region,
        SUM(s.sales) AS total_sales,
        SUM(s.profit) AS total_profit,
        COUNT(DISTINCT o.customer_id) AS customer_count
    FROM 
        sales s
    JOIN 
        orders o ON s.order_id = o.order_id
    GROUP BY 
        o.region
    ORDER BY 
        total_sales DESC
    """
    region_df = get_data_from_db(region_query)
    
    if not region_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(region_df, x='region', y='total_sales',
                        title='Sales by Region',
                        labels={'region': 'Region', 'total_sales': 'Total Sales ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(region_df, values='total_sales', names='region',
                        title='Sales Distribution by Region')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No regional sales data available.")

    # Top States (from dashboard.py)
    st.subheader("🏛️ Top 10 States by Sales")
    state_query = """
    SELECT 
        o.state,
        SUM(s.sales) AS total_sales,
        SUM(s.profit) AS total_profit
    FROM 
        sales s
    JOIN 
        orders o ON s.order_id = o.order_id
    GROUP BY 
        o.state
    ORDER BY 
        total_sales DESC
    LIMIT 10
    """
    state_df = get_data_from_db(state_query)
    
    if not state_df.empty:
        fig = px.bar(state_df, x='total_sales', y='state',
                    orientation='h',
                    title='Top 10 States by Sales',
                    labels={'total_sales': 'Total Sales ($)', 'state': 'State'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No state-level sales data available.")

    # Regional performance with stored procedure (from advanced_dashboard)
    st.subheader("📊 Detailed Regional Performance Analysis")
    
    region_analysis_query = """
    SELECT 
        o.region,
        o.state,
        COUNT(DISTINCT o.customer_id) as unique_customers,
        COUNT(DISTINCT s.order_id) as total_orders,
        SUM(s.sales) as total_sales,
        SUM(s.profit) as total_profit,
        AVG(s.sales) as avg_order_value,
        SUM(s.profit) / SUM(s.sales) * 100 as profit_margin_pct
    FROM sales s
    JOIN orders o ON s.order_id = o.order_id
    GROUP BY o.region, o.state
    ORDER BY total_sales DESC;
    """
    
    regional_detailed_df = get_data_from_db(region_analysis_query)
    if not regional_detailed_df.empty:
        # Regional performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_regions = regional_detailed_df['region'].nunique()
            st.metric("Total Regions", total_regions)
        
        with col2:
            total_states = regional_detailed_df['state'].nunique()
            st.metric("Total States", total_states)
        
        with col3:
            avg_profit_margin = regional_detailed_df['profit_margin_pct'].mean()
            st.metric("Avg Profit Margin", f"{avg_profit_margin:.1f}%")
        
        with col4:
            top_region = regional_detailed_df.groupby('region')['total_sales'].sum().idxmax()
            st.metric("Top Region", top_region)
        
        # Regional visualizations
        region_summary = regional_detailed_df.groupby('region').agg({
            'total_sales': 'sum',
            'total_profit': 'sum',
            'unique_customers': 'sum',
            'total_orders': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(region_summary, x='region', y='total_sales',
                        title='Total Sales by Region')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(region_summary, x='total_sales', y='total_profit',
                           size='unique_customers', hover_name='region',
                           title='Sales vs Profit by Region')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Detailed Regional Data")
        st.dataframe(regional_detailed_df, use_container_width=True)
    else:
        st.info("No detailed regional performance data available.")

def show_sql_query_lab():
    st.header("🔍 SQL Query Laboratory")
    st.markdown("*Advanced SQL execution environment with performance monitoring*")
    
    # Enhanced query templates
    st.subheader("📋 Advanced Query Templates")
    template_options = {
        "Custom Query": "",
        "📊 Sales by Category (View)": "SELECT * FROM sales_by_category_view ORDER BY total_sales DESC;",
        "📈 Monthly Trends (View)": "SELECT * FROM monthly_sales_profit_view ORDER BY sales_month;",
        "🏆 Top Products (Stored Procedure)": "CALL GetTopNProductsBySales(15);",
        "💎 Customer Lifetime Value (CTE)": """
WITH CustomerMetrics AS (
    SELECT 
        o.customer_id,
        o.customer_name,
        SUM(s.sales) AS total_sales,
        COUNT(DISTINCT s.order_id) AS total_orders,
        MIN(o.order_date) AS first_order,
        MAX(o.order_date) AS last_order,
        DATEDIFF(MAX(o.order_date), MIN(o.order_date)) AS customer_lifespan_days
    FROM sales s
    JOIN orders o ON s.order_id = o.order_id
    GROUP BY o.customer_id, o.customer_name
    HAVING total_orders >= 2
),
CustomerSegments AS (
    SELECT 
        customer_name,
        total_sales,
        total_orders,
        customer_lifespan_days,
        (total_sales / total_orders) AS avg_order_value,
        CASE 
            WHEN customer_lifespan_days > 0 
            THEN (total_sales / customer_lifespan_days) * 365 
            ELSE total_sales 
        END AS estimated_annual_value,
        CASE 
            WHEN total_sales >= 5000 THEN 'VIP'
            WHEN total_sales >= 2000 THEN 'High Value'
            WHEN total_sales >= 500 THEN 'Medium Value'
            ELSE 'Low Value'
        END AS customer_segment
    FROM CustomerMetrics
)
SELECT 
    customer_segment,
    COUNT(*) as customer_count,
    AVG(total_sales) as avg_customer_value,
    SUM(total_sales) as segment_total_sales
FROM CustomerSegments
GROUP BY customer_segment
ORDER BY segment_total_sales DESC;
        """,
        "🌍 Regional Performance Analysis": """
SELECT 
    o.region,
    o.state,
    COUNT(DISTINCT o.customer_id) as unique_customers,
    COUNT(DISTINCT s.order_id) as total_orders,
    SUM(s.sales) as total_sales,
    SUM(s.profit) as total_profit,
    AVG(s.sales) as avg_order_value,
    SUM(s.profit) / SUM(s.sales) * 100 as profit_margin_pct,
    RANK() OVER (ORDER BY SUM(s.sales) DESC) as sales_rank
FROM sales s
JOIN orders o ON s.order_id = o.order_id
GROUP BY o.region, o.state
HAVING total_sales > 1000
ORDER BY total_sales DESC
LIMIT 20;
        """,
        "📦 Product Performance with Window Functions": """
SELECT 
    p.category,
    p.sub_category,
    p.product_name,
    SUM(s.sales) as product_sales,
    SUM(s.profit) as product_profit,
    COUNT(*) as units_sold,
    RANK() OVER (PARTITION BY p.category ORDER BY SUM(s.sales) DESC) as category_rank,
    PERCENT_RANK() OVER (ORDER BY SUM(s.sales)) as sales_percentile,
    LAG(SUM(s.sales)) OVER (PARTITION BY p.category ORDER BY SUM(s.sales) DESC) as prev_product_sales
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.category, p.sub_category, p.product_name
HAVING product_sales > 500
ORDER BY p.category, category_rank;
        """,
        "🔄 Sales Cohort Analysis": """
WITH CustomerFirstOrder AS (
    SELECT 
        customer_id,
        MIN(order_date) as first_order_date,
        DATE_FORMAT(MIN(order_date), '%Y-%m') as cohort_month
    FROM orders
    GROUP BY customer_id
),
CustomerOrders AS (
    SELECT 
        o.customer_id,
        o.order_date,
        DATE_FORMAT(o.order_date, '%Y-%m') as order_month,
        cfo.cohort_month,
        PERIOD_DIFF(DATE_FORMAT(o.order_date, '%Y%m'), DATE_FORMAT(cfo.first_order_date, '%Y%m')) as period_number
    FROM orders o
    JOIN CustomerFirstOrder cfo ON o.customer_id = cfo.customer_id
)
SELECT 
    cohort_month,
    period_number,
    COUNT(DISTINCT customer_id) as customers,
    COUNT(DISTINCT customer_id) * 100.0 / 
        FIRST_VALUE(COUNT(DISTINCT customer_id)) OVER (
            PARTITION BY cohort_month 
            ORDER BY period_number 
            ROWS UNBOUNDED PRECEDING
        ) as retention_rate
FROM CustomerOrders
WHERE cohort_month >= '2014-01'
GROUP BY cohort_month, period_number
ORDER BY cohort_month, period_number;
        """
    }
    
    selected_template = st.selectbox("Select a template:", list(template_options.keys()))
    
    # Query input with syntax highlighting
    query = st.text_area(
        "SQL Query:", 
        value=template_options[selected_template],
        height=300,
        help="Enter your SQL query here. Supports SELECT, INSERT, UPDATE, DELETE, and stored procedure calls."
    )
    
    # Query execution controls
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        execute_button = st.button("🚀 Execute Query", type="primary")
    
    with col2:
        if st.button("💾 Save Query"):
            if 'saved_queries' not in st.session_state:
                st.session_state.saved_queries = []
            st.session_state.saved_queries.append({
                'query': query,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'template': selected_template
            })
            st.success("Query saved to history!")
    
    with col3:
        explain_button = st.button("📋 Explain Query")
    
    with col4:
        if st.button("🗑️ Clear"):
            st.session_state.saved_queries = [] # Clear saved queries
            st.rerun()
    
    # Query execution
    if execute_button and query.strip():
        with st.spinner("Executing query..."):
            result_df, execution_time, error_msg = execute_custom_query(query)
            
            if error_msg:
                st.error(f"❌ {error_msg}")
            elif result_df is not None:
                # Success metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("⏱️ Execution Time", f"{execution_time:.3f}s")
                col2.metric("📊 Rows Returned", len(result_df))
                col3.metric("📋 Columns", len(result_df.columns) if not result_df.empty else 0)
                
                if not result_df.empty:
                    # Display results
                    st.subheader("📊 Query Results")
                    st.dataframe(result_df, use_container_width=True, height=400)
                    
                    # Quick visualization options
                    numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                    if len(numeric_cols) > 0 and len(result_df.columns) >= 2:
                        st.subheader("📈 Quick Visualization")
                        
                        viz_col1, viz_col2, viz_col3 = st.columns(3)
                        with viz_col1:
                            chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"])
                        with viz_col2:
                            x_col_options = result_df.columns.tolist()
                            x_col = st.selectbox("X-axis:", x_col_options, key="x_col_select")
                        with viz_col3:
                            y_col_options = numeric_cols
                            y_col = st.selectbox("Y-axis:", y_col_options, key="y_col_select")
                        
                        if st.button("Generate Chart", key="generate_chart_button"):
                            try:
                                if chart_type == "Bar Chart":
                                    fig = px.bar(result_df.head(20), x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                elif chart_type == "Line Chart":
                                    fig = px.line(result_df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                                elif chart_type == "Scatter Plot":
                                    fig = px.scatter(result_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                                elif chart_type == "Pie Chart":
                                    # For pie chart, values should be numeric, names can be any column
                                    if y_col: # Ensure y_col is selected and numeric
                                        fig = px.pie(result_df.head(10), values=y_col, names=x_col, title=f"{y_col} Distribution by {x_col}")
                                    else:
                                        st.warning("Please select a numeric column for Y-axis for Pie Chart.")
                                        fig = None
                                
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Visualization error: {str(e)}")
                    
                    # Export options
                    st.subheader("📥 Export Options")
                    export_col1, export_col2, export_col3 = st.columns(3)
                    
                    with export_col1:
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with export_col2:
                        json_data = result_df.to_json(orient='records', indent=2).encode('utf-8')
                        st.download_button(
                            label="📋 Download JSON",
                            data=json_data,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with export_col3:
                        # Use BytesIO for Excel export
                        from io import BytesIO
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            result_df.to_excel(writer, index=False, sheet_name='Query Results')
                        excel_buffer.seek(0)
                        st.download_button(
                            label="📊 Download Excel",
                            data=excel_buffer,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.info("Query executed successfully but returned no results.")
            else:
                st.success(f"✅ Query executed successfully in {execution_time:.3f} seconds (Non-SELECT query).")
    
    # Query explanation
    if explain_button and query.strip():
        st.subheader("📋 Query Explanation")
        # Only explain SELECT queries
        if query.strip().upper().startswith('SELECT'):
            explain_query = f"EXPLAIN {query}"
            explain_df, _, explain_error = execute_custom_query(explain_query)
            
            if explain_error:
                st.error(f"Cannot explain query: {explain_error}")
            elif explain_df is not None and not explain_df.empty:
                st.dataframe(explain_df, use_container_width=True)
            else:
                st.info("EXPLAIN query returned no results or encountered an issue.")
        else:
            st.warning("EXPLAIN is only applicable for SELECT statements.")
    
    # Query History
    if 'saved_queries' in st.session_state and st.session_state.saved_queries:
        st.subheader("📚 Query History")
        
        # Display in reverse chronological order, limit to last 10
        for i, saved_query in enumerate(reversed(st.session_state.saved_queries[-10:])):
            with st.expander(f"🔍 {saved_query['template']} - {saved_query['timestamp']}"):
                st.code(saved_query['query'], language='sql')
                # Use a unique key for each button
                if st.button(f"🔄 Load Query", key=f"load_history_query_{i}"):
                    # This will reload the page and set the text area value
                    st.session_state.query_text_area = saved_query['query']
                    st.session_state.selected_template_box = "Custom Query" # Reset template selection
                    st.rerun()

# This is a workaround to pre-fill the text area after a rerun
if 'query_text_area' not in st.session_state:
    st.session_state.query_text_area = ""
if 'selected_template_box' not in st.session_state:
    st.session_state.selected_template_box = "Custom Query"

def show_advanced_analytics():
    st.header("📈 Advanced Analytics")
    
    # Customer Lifetime Value Analysis (already integrated into Customer Insights)
    st.subheader("💎 Customer Lifetime Value Analysis (Detailed)")
    st.info("This section provides a more detailed view of Customer Lifetime Value. A summary is available under 'Customer Insights'.")
    
    clv_query = """
    WITH CustomerMetrics AS (
        SELECT 
            o.customer_id,
            o.customer_name,
            SUM(s.sales) AS total_sales,
            COUNT(DISTINCT s.order_id) AS total_orders,
            MIN(o.order_date) AS first_order,
            MAX(o.order_date) AS last_order,
            DATEDIFF(MAX(o.order_date), MIN(o.order_date)) AS customer_lifespan_days
        FROM sales s
        JOIN orders o ON s.order_id = o.order_id
        GROUP BY o.customer_id, o.customer_name
        HAVING total_orders >= 2
    ),
    CustomerSegments AS (
        SELECT 
            customer_name,
            total_sales,
            total_orders,
            customer_lifespan_days,
            (total_sales / total_orders) AS avg_order_value,
            CASE 
                WHEN customer_lifespan_days > 0 
                THEN (total_sales / customer_lifespan_days) * 365 
                ELSE total_sales 
            END AS estimated_annual_value,
            CASE 
                WHEN total_sales >= 5000 THEN 'VIP'
                WHEN total_sales >= 2000 THEN 'High Value'
                WHEN total_sales >= 500 THEN 'Medium Value'
                ELSE 'Low Value'
            END AS customer_segment
        FROM CustomerMetrics
    )
    SELECT 
        customer_name,
        total_sales,
        total_orders,
        avg_order_value,
        customer_lifespan_days,
        estimated_annual_value,
        customer_segment
    FROM CustomerSegments
    ORDER BY estimated_annual_value DESC
    LIMIT 50;
    """
    
    clv_df = get_data_from_db(clv_query)
    if not clv_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                clv_df, 
                x='avg_order_value', 
                y='estimated_annual_value',
                size='total_orders',
                color='customer_segment',
                hover_name='customer_name',
                title='Customer Value Analysis by Segment'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(clv_df[['customer_name', 'customer_segment', 'total_sales', 'estimated_annual_value', 'avg_order_value']], use_container_width=True)
    else:
        st.info("No detailed customer lifetime value data available.")

    # Cohort Analysis (from advanced_dashboard)
    st.subheader("👥 Customer Cohort Analysis")
    
    cohort_query = """
    WITH CustomerFirstOrder AS (
        SELECT 
            customer_id,
            MIN(order_date) as first_order_date,
            DATE_FORMAT(MIN(order_date), '%Y-%m') as cohort_month
        FROM orders
        GROUP BY customer_id
    ),
    CustomerOrders AS (
        SELECT 
            o.customer_id,
            o.order_date,
            DATE_FORMAT(o.order_date, '%Y-%m') as order_month,
            cfo.cohort_month,
            PERIOD_DIFF(DATE_FORMAT(o.order_date, '%Y%m'), DATE_FORMAT(cfo.first_order_date, '%Y%m')) as period_number
        FROM orders o
        JOIN CustomerFirstOrder cfo ON o.customer_id = cfo.customer_id
    )
    SELECT 
        cohort_month,
        period_number,
        COUNT(DISTINCT customer_id) as customers,
        COUNT(DISTINCT customer_id) * 100.0 / 
            FIRST_VALUE(COUNT(DISTINCT customer_id)) OVER (
                PARTITION BY cohort_month 
                ORDER BY period_number 
                ROWS UNBOUNDED PRECEDING
            ) as retention_rate
    FROM CustomerOrders
    WHERE cohort_month >= '2014-01' -- Filter to avoid very early, sparse cohorts
    GROUP BY cohort_month, period_number
    ORDER BY cohort_month, period_number;
    """
    
    cohort_df = get_data_from_db(cohort_query)
    if not cohort_df.empty:
        # Create cohort heatmap
        # Ensure 'period_number' is treated as a string for columns if it's not sequential
        cohort_pivot = cohort_df.pivot(index='cohort_month', columns='period_number', values='retention_rate')
        
        # Sort columns numerically if they represent periods
        cohort_pivot = cohort_pivot.reindex(columns=sorted(cohort_pivot.columns))

        fig = px.imshow(
            cohort_pivot.values,
            x=cohort_pivot.columns,
            y=cohort_pivot.index,
            title='Customer Retention Heatmap (%)',
            color_continuous_scale='Blues',
            text_auto=True # Show values on heatmap
        )
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cohort_df, use_container_width=True)
    else:
        st.info("No customer cohort data available.")

def show_predictive_forecasting():
    st.header("🔮 Predictive Sales Forecasting")
    st.markdown("*Advanced time series forecasting using multiple algorithms*")
    
    # Forecasting controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_periods = st.slider("Forecast Periods (Months):", min_value=1, max_value=12, value=6)
    
    with col2:
        forecast_metric = st.selectbox("Forecast Metric:", ["Sales", "Profit", "Orders"])
    
    with col3:
        if st.button("🚀 Generate Forecasts", type="primary"):
            with st.spinner("Generating forecasts..."):
                try:
                    # Pass the selected metric to the forecasting function
                    forecasts_df, accuracy_metrics, historical_df = get_sales_forecasts(periods=forecast_periods, target_column=f'monthly_{forecast_metric.lower()}')
                    
                    if not forecasts_df.empty and not historical_df.empty:
                        st.success("✅ Forecasts generated successfully!")
                        
                        # Display accuracy metrics
                        if accuracy_metrics:
                            st.subheader("🎯 Model Accuracy (MAPE)")
                            acc_col1, acc_col2, acc_col3, acc_col4 = st.columns(4)
                            
                            metrics_list = list(accuracy_metrics.items())
                            for i, (model, accuracy) in enumerate(metrics_list):
                                if i == 0:
                                    acc_col1.metric(f"📈 {model}", accuracy)
                                elif i == 1:
                                    acc_col2.metric(f"📊 {model}", accuracy)
                                elif i == 2:
                                    acc_col3.metric(f"📉 {model}", accuracy)
                                elif i == 3:
                                    acc_col4.metric(f"🔄 {model}", accuracy)
                        
                        # Visualization
                        st.subheader(f"📈 {forecast_metric} Forecast Visualization")
                        
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=historical_df['month_date'],
                            y=historical_df[f'monthly_{forecast_metric.lower()}'],
                            mode='lines+markers',
                            name=f'Historical {forecast_metric}',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Forecasts by method
                        colors = ['red', 'green', 'orange', 'purple']
                        for i, method in enumerate(forecasts_df['forecast_type'].unique()):
                            method_data = forecasts_df[forecasts_df['forecast_type'] == method]
                            fig.add_trace(go.Scatter(
                                x=method_data['month_date'],
                                y=method_data['predicted_value'],
                                mode='lines+markers',
                                name=f'{method} Forecast',
                                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                            ))
                        
                        fig.update_layout(
                            title=f'{forecast_metric} Forecasting - Multiple Methods Comparison',
                            xaxis_title='Date',
                            yaxis_title=f'{forecast_metric} ($)' if forecast_metric == "Sales" or forecast_metric == "Profit" else forecast_metric,
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast table
                        st.subheader("📊 Detailed Forecast Results")
                        
                        # Pivot the forecasts for better display
                        forecast_pivot = forecasts_df.pivot(index='month_date', columns='forecast_type', values='predicted_value')
                        forecast_pivot = forecast_pivot.round(2)
                        forecast_pivot.index = forecast_pivot.index.strftime('%Y-%m')
                        
                        st.dataframe(forecast_pivot, use_container_width=True)
                        
                        # Download forecast results
                        csv_data = forecast_pivot.to_csv().encode('utf-8')
                        st.download_button(
                            label="📥 Download Forecast Data",
                            data=csv_data,
                            file_name=f"{forecast_metric.lower()}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        
                    else:
                        st.error("❌ Unable to generate forecasts. Insufficient historical data or an issue occurred.")
                        
                except Exception as e:
                    st.error(f"❌ Forecasting error: {str(e)}")
    
    # Forecasting methodology explanation
    with st.expander("📚 Forecasting Methodology"):
        st.markdown("""
        ### 🔬 Forecasting Methods Used:
        
        1.  **Linear Regression**: Simple trend-based forecasting.
        2.  **Polynomial Regression**: Captures non-linear patterns.
        3.  **Moving Average**: Smooths out short-term fluctuations.
        4.  **Seasonal Forecasting**: Accounts for seasonal patterns.
        
        ### 📊 Accuracy Metrics:
        -   **MAPE (Mean Absolute Percentage Error)**: Lower values indicate better accuracy.
            \$$
            MAPE = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{A_t - F_t}{A_t} \right| \times 100\%
            \$$
            Where:
            -   \$A_t\$ is the actual value.
            -   \$F_t\$ is the forecast value.
            -   \$n\$ is the number of data points.
        -   Models are validated using the last 6 months of historical data.
        
        ### ⚠️ Important Notes:
        -   Forecasts are estimates based on historical patterns.
        -   External factors (market changes, economic conditions) may affect actual results.
        -   Use multiple methods for better decision-making.
        """)

def show_performance_dashboard():
    st.header("⚡ Performance Dashboard")
    st.markdown("*Database performance monitoring and optimization insights*")
    
    # Database statistics
    st.subheader("📊 Database Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Table sizes and statistics
    table_stats_query = """
    SELECT 
        table_name,
        table_rows,
        ROUND(((data_length + index_length) / 1024 / 1024), 2) AS size_mb,
        ROUND((data_length / 1024 / 1024), 2) AS data_mb,
        ROUND((index_length / 1024 / 1024), 2) AS index_mb
    FROM information_schema.tables 
    WHERE table_schema = 'retail_sales'
    ORDER BY size_mb DESC;
    """
    
    table_stats_df = get_data_from_db(table_stats_query)
    if not table_stats_df.empty:
        with col1:
            st.metric("📋 Total Tables", len(table_stats_df))
        with col2:
            total_rows = table_stats_df['table_rows'].sum()
            st.metric("📊 Total Rows", f"{total_rows:,}")
        with col3:
            total_size = table_stats_df['size_mb'].sum()
            st.metric("💾 Database Size", f"{total_size:.2f} MB")
        with col4:
            avg_size = table_stats_df['size_mb'].mean()
            st.metric("📈 Avg Table Size", f"{avg_size:.2f} MB")
        
        # Table details
        st.subheader("📋 Table Statistics")
        st.dataframe(table_stats_df, use_container_width=True)
    else:
        st.info("No database table statistics available.")
    
    # Query performance testing
    st.subheader("🏃‍♂️ Query Performance Benchmarks")
    
    performance_queries = {
        "Simple Count": "SELECT COUNT(*) FROM sales;",
        "Basic Join": "SELECT COUNT(*) FROM sales s JOIN orders o ON s.order_id = o.order_id;",
        "Complex Aggregation": """
            SELECT region, category, SUM(sales), AVG(profit) 
            FROM sales s 
            JOIN orders o ON s.order_id = o.order_id 
            JOIN products p ON s.product_id = p.product_id 
            GROUP BY region, category;
        """,
        "View Query": "SELECT COUNT(*) FROM sales_by_category_view;",
        "Stored Procedure": "CALL GetTopNProductsBySales(10);",
        "Window Function": """
            SELECT customer_name, sales, 
                   ROW_NUMBER() OVER (ORDER BY sales DESC) as rank
            FROM sales s JOIN orders o ON s.order_id = o.order_id 
            LIMIT 100;
        """,
        "CTE Query": """
            WITH monthly_sales AS (
                SELECT DATE_FORMAT(order_date, '%Y-%m') as month, SUM(sales) as total
                FROM sales s JOIN orders o ON s.order_id = o.order_id
                GROUP BY month
            )
            SELECT COUNT(*) FROM monthly_sales;
        """
    }
    
    if st.button("🚀 Run Performance Tests"):
        results = []
        progress_bar = st.progress(0)
        
        for i, (query_name, query) in enumerate(performance_queries.items()):
            with st.spinner(f"Testing {query_name}..."):
                _, execution_time, error = execute_custom_query(query)
                
                # Categorize performance
                if execution_time < 0.1:
                    performance_category = "🟢 Excellent"
                elif execution_time < 0.5:
                    performance_category = "🟡 Good"
                elif execution_time < 1.0:
                    performance_category = "🟠 Fair"
                else:
                    performance_category = "🔴 Slow"
                
                results.append({
                    'Query Type': query_name,
                    'Execution Time (s)': f"{execution_time:.4f}",
                    'Performance': performance_category,
                    'Status': '✅ Success' if not error else '❌ Error'
                })
                
                progress_bar.progress((i + 1) / len(performance_queries))
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Performance visualization
        if not results_df.empty:
            results_df['Execution Time (numeric)'] = results_df['Execution Time (s)'].astype(float)
            fig = px.bar(
                results_df, 
                x='Query Type', 
                y='Execution Time (numeric)',
                title='Query Performance Comparison',
                color='Execution Time (numeric)',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click 'Run Performance Tests' to benchmark common queries.")

def show_data_export_center():
    st.header("📋 Data Export Center")
    st.markdown("*Comprehensive data export and reporting capabilities*")
    
    # Export options
    st.subheader("📊 Available Data Exports")
    
    export_options = {
        "📈 Sales Summary Report": """
            SELECT 
                DATE_FORMAT(o.order_date, '%Y-%m') as month,
                SUM(s.sales) as total_sales,
                SUM(s.profit) as total_profit,
                COUNT(DISTINCT s.order_id) as orders,
                COUNT(DISTINCT o.customer_id) as customers
            FROM sales s
            JOIN orders o ON s.order_id = o.order_id
            GROUP BY month
            ORDER BY month;
        """,
        "👤 Customer Analysis Report": """
            SELECT 
                o.customer_name,
                o.segment,
                o.region,
                COUNT(DISTINCT s.order_id) as total_orders,
                SUM(s.sales) as total_sales,
                SUM(s.profit) as total_profit,
                AVG(s.sales) as avg_order_value,
                MIN(o.order_date) as first_order,
                MAX(o.order_date) as last_order
            FROM sales s
            JOIN orders o ON s.order_id = o.order_id
            GROUP BY o.customer_id, o.customer_name, o.segment, o.region
            ORDER BY total_sales DESC;
        """,
        "📦 Product Performance Report": """
            SELECT 
                p.category,
                p.sub_category,
                p.product_name,
                SUM(s.sales) as total_sales,
                SUM(s.profit) as total_profit,
                SUM(s.quantity) as units_sold,
                AVG(s.discount) as avg_discount,
                SUM(s.profit) / SUM(s.sales) * 100 as profit_margin
            FROM sales s
            JOIN products p ON s.product_id = p.product_id
            GROUP BY p.product_id, p.category, p.sub_category, p.product_name
            ORDER BY total_sales DESC;
        """,
        "🌍 Regional Performance Report": """
            SELECT 
                o.region,
                o.state,
                o.city,
                COUNT(DISTINCT o.customer_id) as customers,
                COUNT(DISTINCT s.order_id) as orders,
                SUM(s.sales) as total_sales,
                SUM(s.profit) as total_profit,
                AVG(s.sales) as avg_order_value
            FROM sales s
            JOIN orders o ON s.order_id = o.order_id
            GROUP BY o.region, o.state, o.city
            ORDER BY total_sales DESC;
        """,
        "📊 Complete Dataset": """
            SELECT 
                s.row_id,
                o.order_id,
                o.order_date,
                o.ship_date,
                o.ship_mode,
                o.customer_id,
                o.customer_name,
                o.segment,
                o.country,
                o.city,
                o.state,
                o.postal_code,
                o.region,
                p.product_id,
                p.category,
                p.sub_category,
                p.product_name,
                s.sales,
                s.quantity,
                s.discount,
                s.profit
            FROM sales s
            JOIN orders o ON s.order_id = o.order_id
            JOIN products p ON s.product_id = p.product_id
            ORDER BY o.order_date, s.row_id;
        """
    }
    
    selected_export = st.selectbox("Select Report Type:", list(export_options.keys()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox("Export Format:", ["CSV", "Excel", "JSON"])
    
    with col2:
        include_timestamp = st.checkbox("Include Timestamp in Filename", value=True)
    
    if st.button("📥 Generate Export", type="primary"):
        with st.spinner("Generating export..."):
            try:
                export_df, execution_time, error = execute_custom_query(export_options[selected_export])
                
                if error:
                    st.error(f"❌ Export failed: {error}")
                elif export_df is not None and not export_df.empty:
                    st.success(f"✅ Export generated successfully! ({len(export_df)} rows in {execution_time:.3f}s)")
                    
                    # Preview
                    st.subheader("📊 Data Preview")
                    st.dataframe(export_df.head(100), use_container_width=True)
                    
                    # Generate filename
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if include_timestamp else ''
                    # Clean up selected_export string for filename
                    base_name = selected_export.replace('📈 ', '').replace('👤 ', '').replace('📦 ', '').replace('🌍 ', '').replace('📊 ', '').replace(' ', '_').lower()
                    
                    # Export based on format
                    if export_format == "CSV":
                        filename = f"{base_name}_{timestamp}.csv" if timestamp else f"{base_name}.csv"
                        data = export_df.to_csv(index=False).encode('utf-8')
                        mime_type = "text/csv"
                    elif export_format == "Excel":
                        filename = f"{base_name}_{timestamp}.xlsx" if timestamp else f"{base_name}.xlsx"
                        from io import BytesIO
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            export_df.to_excel(writer, index=False, sheet_name='Exported Data')
                        excel_buffer.seek(0)
                        data = excel_buffer
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    else:  # JSON
                        filename = f"{base_name}_{timestamp}.json" if timestamp else f"{base_name}.json"
                        data = export_df.to_json(orient='records', indent=2).encode('utf-8')
                        mime_type = "application/json"
                    
                    st.download_button(
                        label=f"📥 Download {export_format}",
                        data=data,
                        file_name=filename,
                        mime=mime_type
                    )
                    
                    # Export summary
                    st.subheader("📋 Export Summary")
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    summary_col1.metric("📊 Total Rows", len(export_df))
                    summary_col2.metric("📋 Total Columns", len(export_df.columns))
                    summary_col3.metric("⏱️ Generation Time", f"{execution_time:.3f}s")
                    
                else:
                    st.warning("⚠️ Export completed but no data was returned.")
                    
            except Exception as e:
                st.error(f"❌ Export error: {str(e)}")

if __name__ == "__main__":
    main()
