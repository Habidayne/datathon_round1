SELECT 
    payment_method, 
    COUNT(*) AS cancel_count
FROM orders
WHERE order_status = 'cancelled'
GROUP BY payment_method
ORDER BY cancel_count DESC
LIMIT 1;