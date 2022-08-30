SELECT neighborhood, AVG(price_per_sqft) AS "average price per square feet", 
                    AVG(overallqual) AS "overall quality" 
FROM houses
GROUP BY neighborhood
ORDER BY AVG(price_per_sqft) desc;

