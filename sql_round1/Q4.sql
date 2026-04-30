SELECT traffic_source, AVG(bounce_rate) AS avg_bounce_rate
FROM web_traffic
GROUP BY traffic_source
ORDER BY avg_bounce_rate ASC
LIMIT 1;