-- creates a stored procedure AddBonus that adds a new correction for a student.
DELIMITER //

CREATE PROCEDURE AddBonus (IN user_id INT, IN project_name VARCHAR(255), IN score INT)
BEGIN
    DECLARE proje_id INT;
    
    IF NOT EXISTS (SELECT id from projects WHERE name = project_name) THEN
        INSERT INTO projects (name) VALUES (project_name);
    END IF;
    SELECT id INTO proje_id FROM projects WHERE name = project_name;

    INSERT INTO corrections (user_id, project_id, score)
    VALUES (user_id, proje_id, score);
END;
//

DELIMITER ;
