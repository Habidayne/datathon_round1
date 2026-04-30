SELECT 
    g.region, 
    SUM((oi.quantity * oi.unit_price) - oi.discount_amount) AS total_revenue
FROM geography g
JOIN orders o ON g.zip = o.zip
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY g.region
ORDER BY total_revenue DESC
LIMIT 1;