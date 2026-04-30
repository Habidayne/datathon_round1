WITH OrderDates AS (
    SELECT 
        customer_id,
        order_date,
        LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_order_date
    FROM orders
),
Gaps AS (
    SELECT DATEDIFF(order_date, prev_order_date) AS gap_days
    FROM OrderDates
    WHERE prev_order_date IS NOT NULL
)
-- Khối lệnh tính trung vị thủ công cho MySQL
SELECT AVG(gap_days) AS median_gap
FROM (
    SELECT gap_days, 
           ROW_NUMBER() OVER (ORDER BY gap_days) AS row_num,
           COUNT(*) OVER () AS total_rows
    FROM Gaps
) sub
WHERE row_num IN (FLOOR((total_rows + 1) / 2), CEIL((total_rows + 1) / 2));