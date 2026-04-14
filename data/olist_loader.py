# =========================================
# OLIST DATA LOADER
# Creates SQLite Database
# =========================================

import pandas as pd
import sqlite3
import os

# ================================
# FILE PATHS
# ================================

DATA_PATH = "data/raw"

orders_path = os.path.join(DATA_PATH, "olist_orders_dataset.csv")
order_items_path = os.path.join(DATA_PATH, "olist_order_items_dataset.csv")
products_path = os.path.join(DATA_PATH, "olist_products_dataset.csv")
reviews_path = os.path.join(DATA_PATH, "olist_order_reviews_dataset.csv")
customers_path = os.path.join(DATA_PATH, "olist_customers_dataset.csv")
sellers_path = os.path.join(DATA_PATH, "olist_sellers_dataset.csv")
payments_path = os.path.join(DATA_PATH, "olist_order_payments_dataset.csv")
geo_path = os.path.join(DATA_PATH, "olist_geolocation_dataset.csv")

# ================================
# LOAD CSV FILES
# ================================

print("Loading datasets...")

orders = pd.read_csv(orders_path)
order_items = pd.read_csv(order_items_path)
products = pd.read_csv(products_path)
reviews = pd.read_csv(reviews_path)
customers = pd.read_csv(customers_path)
sellers = pd.read_csv(sellers_path)
payments = pd.read_csv(payments_path)
geo = pd.read_csv(geo_path)

print("Datasets loaded successfully!")

# ================================
# CLEAN REVIEW TEXT
# ================================

print("Cleaning review text...")

reviews["review_comment_message"] = (
    reviews["review_comment_message"]
    .fillna("")
    .astype(str)
)

# ================================
# CREATE MERGED ANALYTICS VIEW
# ================================

print("Creating analytics view...")

df = orders.merge(
    order_items,
    on="order_id",
    how="left"
)

df = df.merge(
    products,
    on="product_id",
    how="left"
)

df = df.merge(
    reviews,
    on="order_id",
    how="left"
)

df = df.merge(
    customers,
    on="customer_id",
    how="left"
)

df = df.merge(
    sellers,
    on="seller_id",
    how="left"
)

df = df.merge(
    payments,
    on="order_id",
    how="left"
)

print("Merged dataset shape:", df.shape)

# ================================
# CREATE SQLITE DATABASE
# ================================

print("Creating SQLite database...")

conn = sqlite3.connect("data/ecommerce.db")

orders.to_sql("orders", conn, if_exists="replace", index=False)
order_items.to_sql("order_items", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
reviews.to_sql("reviews", conn, if_exists="replace", index=False)
customers.to_sql("customers", conn, if_exists="replace", index=False)
sellers.to_sql("sellers", conn, if_exists="replace", index=False)
payments.to_sql("payments", conn, if_exists="replace", index=False)
geo.to_sql("geolocation", conn, if_exists="replace", index=False)

df.to_sql("analytics_view", conn, if_exists="replace", index=False)

conn.close()

print("✅ SQLite database created successfully!")
print("Database saved at: data/ecommerce.db")