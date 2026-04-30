SELECT segment, AVG((price - cogs) / price) AS avg_gross_margin
FROM products
GROUP BY segment
ORDER BY avg_gross_margin DESC
LIMIT 1;