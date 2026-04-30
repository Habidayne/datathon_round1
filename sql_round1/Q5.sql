SELECT 
    (COUNT(CASE WHEN promo_id IS NOT NULL THEN 1 END) * 100.0) / COUNT(*) AS promo_percentage
FROM order_items;