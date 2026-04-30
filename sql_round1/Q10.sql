SELECT 
    installments, 
    AVG(payment_value) AS avg_payment_per_order
FROM payments
GROUP BY installments
ORDER BY avg_payment_per_order DESC
LIMIT 1;