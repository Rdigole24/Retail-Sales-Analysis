# Retail Sales Dashboard

A comprehensive retail sales analytics dashboard built with Python, Streamlit, and MySQL. This application provides interactive visualizations and insights for retail sales data analysis.

## 🚀 Features

- **Interactive Dashboard**: Real-time sales analytics with Streamlit
- **Data Visualization**: Charts and graphs using Matplotlib, Seaborn, and Plotly
- **Database Integration**: MySQL database for efficient data storage and retrieval
- **ETL Pipeline**: Automated data extraction, transformation, and loading
- **Sales Analytics**: Key metrics including revenue, trends, and performance indicators

## 🛠️ Technologies Used

- **Backend**: Python 3.7+
- **Database**: MySQL
- **Frontend**: Streamlit
- **Data Processing**: Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Database Connector**: mysql-connector-python

## 📋 Prerequisites

Before running this application, ensure you have the following installed:

- Python 3.7 or higher
- MySQL Server
- Git (optional, for cloning)

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
# Using Git
git clone <repository-url>
cd retail_sales_dashboard

# Or download and extract the project files manually
```

### 2. Install Python Dependencies

```bash
pip install pandas mysql-connector-python streamlit matplotlib seaborn plotly
```

Alternatively, if you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Setup MySQL Database

Start your MySQL service:

```bash
# On Linux/Mac
sudo service mysql start

# On Windows
net start mysql
```

Configure MySQL root user (if needed):

```bash
sudo mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY ''; FLUSH PRIVILEGES;"
```

### 4. Run the ETL Pipeline

Execute the ETL script to set up the database and load data:

```bash
python3 etl_script.py
```

This script will:
- Create the `retail_sales` database
- Create necessary tables (`orders`, `products`, `sales`)
- Load data from CSV files into the database

### 5. Launch the Dashboard

Start the Streamlit application:

```bash
streamlit run dashboard.py
```

The dashboard will be available at: **http://localhost:8501**

## 🗄️ Database Schema

The application creates the following tables:

- **orders**: Order information and metadata
- **products**: Product catalog and details
- **sales**: Sales transactions and metrics

## 📊 Dashboard Features

- **Sales Overview**: Key performance indicators and summary metrics
- **Revenue Trends**: Time-series analysis of sales performance
- **Product Analysis**: Top-selling products and category insights
- **Interactive Filters**: Date range, product categories, and region filters
- **Export Capabilities**: Download reports and visualizations

## 🚨 Troubleshooting

### Common Issues

1. **MySQL Connection Error**
   ```
   Error: Access denied for user 'root'@'localhost'
   ```
   **Solution**: Ensure MySQL is running and credentials are correct

2. **Module Not Found Error**
   ```
   ModuleNotFoundError: No module named 'streamlit'
   ```
   **Solution**: Install required packages using pip

3. **Port Already in Use**
   ```
   Error: Port 8501 is already in use
   ```
   **Solution**: Use a different port: `streamlit run dashboard.py --server.port 8502`

### Database Issues

If you encounter database connection issues:

```bash
# Check MySQL service status
sudo service mysql status

# Restart MySQL service
sudo service mysql restart
```

## 🔒 Configuration

### Database Configuration

Update database credentials in your ETL script if needed:

```python
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Update if you have a password
    'database': 'retail_sales'
}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

If you encounter any issues or have questions:

- Create an issue in this repository
- Ensure all prerequisites are properly installed

## 🎯 Future Enhancements

- [ ] User authentication and role-based access
- [ ] Real-time data streaming
- [ ] Advanced ML analytics and forecasting
- [ ] Mobile-responsive design
- [ ] API endpoints for data access
- [ ] Docker containerization
