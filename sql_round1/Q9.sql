WITH OrderItemCounts AS (
    SELECT p.size, COUNT(oi.line_id) AS total_ordered_lines
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    WHERE p.size IN ('S', 'M', 'L', 'XL')
    GROUP BY p.size
),
ReturnCounts AS (
    SELECT p.size, COUNT(r.return_id) AS total_returned_lines
    FROM returns r
    JOIN products p ON r.product_id = p.product_id
    WHERE p.size IN ('S', 'M', 'L', 'XL')
    GROUP BY p.size
)
SELECT 
    o.size, 
    COALESCE(r.total_returned_lines, 0) / o.total_ordered_lines AS return_rate
FROM OrderItemCounts o
LEFT JOIN ReturnCounts r ON o.size = r.size
ORDER BY return_rate DESC
LIMIT 1;