SELECT 
    c.age_group, 
    COUNT(o.order_id) / COUNT(DISTINCT c.customer_id) AS avg_orders_per_customer
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE c.age_group IS NOT NULL
GROUP BY c.age_group
ORDER BY avg_orders_per_customer DESC
LIMIT 1;