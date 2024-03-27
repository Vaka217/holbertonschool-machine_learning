--  lists all shows from hbtn_0d_tvshows_rate by their rating.
SELECT count(tsr.rate) AS rating
FROM tv_show_ratings tsr
GROUP BY tsr.show_id