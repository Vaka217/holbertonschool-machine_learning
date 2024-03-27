-- lists all genres in the database hbtn_0d_tvshows_rate by their rating.
SELECT tg.name, SUM(tsr.rate) AS rating
FROM tv_genres tg, tv_show_ratings tsr, tv_show_genres tsg
WHERE tsr.show_id = tsg.show_id AND tg.id = tsg.genre_id
GROUP BY tg.name
ORDER BY rating DESC