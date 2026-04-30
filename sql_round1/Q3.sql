SELECT r.return_reason, COUNT(*) as reason_count
FROM returns r
JOIN products p ON r.product_id = p.product_id
WHERE p.category = 'Streetwear'
GROUP BY r.return_reason
ORDER BY reason_count DESC
LIMIT 1;