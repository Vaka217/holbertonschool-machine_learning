-- lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
SELECT tg.name AS genre, count(tsg.show_id) AS number_of_shows 
FROM tv_show_genres tsg, tv_genres tg
WHERE tg.id = tsg.genre_id
GROUP BY tsg.genre_id, genre
ORDER BY number_of_shows DESC;
