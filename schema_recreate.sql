SET FOREIGN_KEY_CHECKS = 0;

-- Xóa bảng cũ nếu tồn tại
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS returns;
DROP TABLE IF EXISTS shipments;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS promotions;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS geography;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS sample_submission;
DROP TABLE IF EXISTS web_traffic;

-- Tạo bảng mới với cấu trúc chuẩn
CREATE TABLE geography (
    zip INT PRIMARY KEY,
    city VARCHAR(255),
    region VARCHAR(255),
    district VARCHAR(255)
);

CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    zip INT,
    city VARCHAR(255),
    signup_date DATE,
    gender VARCHAR(50),
    age_group VARCHAR(50),
    acquisition_channel VARCHAR(255),
    FOREIGN KEY (zip) REFERENCES geography(zip)
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(255),
    category VARCHAR(255),
    segment VARCHAR(255),
    size VARCHAR(50),
    color VARCHAR(50),
    price FLOAT,
    cogs FLOAT,
    CONSTRAINT chk_cogs_price CHECK (cogs < price)
);

CREATE TABLE promotions (
    promo_id VARCHAR(50) PRIMARY KEY,
    promo_name VARCHAR(255),
    promo_type VARCHAR(50),
    discount_value FLOAT,
    start_date DATE,
    end_date DATE,
    applicable_category VARCHAR(255),
    promo_channel VARCHAR(255),
    stackable_flag INT,
    min_order_value FLOAT
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    order_date DATE,
    customer_id INT,
    zip INT,
    order_status VARCHAR(50),
    payment_method VARCHAR(50),
    device_type VARCHAR(50),
    order_source VARCHAR(255),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (zip) REFERENCES geography(zip)
);

CREATE TABLE order_items (
    line_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    unit_price FLOAT,
    discount_amount FLOAT,
    promo_id VARCHAR(50),
    promo_id_2 VARCHAR(50),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (promo_id) REFERENCES promotions(promo_id),
    FOREIGN KEY (promo_id_2) REFERENCES promotions(promo_id)
);

CREATE TABLE payments (
    order_id INT PRIMARY KEY,
    payment_method VARCHAR(50),
    payment_value FLOAT,
    installments INT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE shipments (
    order_id INT PRIMARY KEY,
    ship_date DATE,
    delivery_date DATE,
    shipping_fee FLOAT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE returns (
    return_id VARCHAR(50) PRIMARY KEY,
    order_id INT,
    product_id INT,
    return_date DATE,
    return_reason VARCHAR(255),
    return_quantity INT,
    refund_amount FLOAT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE reviews (
    review_id VARCHAR(50) PRIMARY KEY,
    order_id INT,
    product_id INT,
    customer_id INT,
    review_date DATE,
    rating INT,
    review_title VARCHAR(255),
    CONSTRAINT chk_rating CHECK (rating >= 1 AND rating <= 5),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE inventory (
    snapshot_date DATE,
    product_id INT,
    stock_on_hand INT,
    units_received INT,
    units_sold INT,
    stockout_days INT,
    days_of_supply FLOAT,
    fill_rate FLOAT,
    stockout_flag INT,
    overstock_flag INT,
    reorder_flag INT,
    sell_through_rate FLOAT,
    product_name VARCHAR(255),
    category VARCHAR(255),
    segment VARCHAR(255),
    year INT,
    month INT,
    PRIMARY KEY (snapshot_date, product_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE sales (
    Date DATE PRIMARY KEY,
    Revenue FLOAT,
    COGS FLOAT
);

CREATE TABLE sample_submission (
    Date DATE PRIMARY KEY,
    Revenue FLOAT,
    COGS FLOAT
);

CREATE TABLE web_traffic (
    date DATE,
    sessions INT,
    unique_visitors INT,
    page_views INT,
    bounce_rate FLOAT,
    avg_session_duration_sec FLOAT,
    conversion_rate FLOAT,
    traffic_source VARCHAR(255),
    PRIMARY KEY (date, traffic_source)
);

SET FOREIGN_KEY_CHECKS = 1;