-- creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id_p INT)
BEGIN
    DECLARE avg_score INT;
    
    SELECT AVG(score) INTO avg_score FROM corrections WHERE user_id = user_id_p;

    UPDATE users SET average_score = avg_score WHERE id = user_id_p;
END;
//

DELIMITER ;
