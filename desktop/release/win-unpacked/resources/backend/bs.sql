CREATE DATABASE bs;
USE bs;
CREATE TABLE images(
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255),
    filepath VARCHAR(255),
    uoload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);
CREATE TABLE detections(
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT,
    label VARCHAR(100),
    confidence FLOAT,
    x_center INT,
    y_center INT,
    weight INT,
    height INT,
    FOREIGN KEY (image_id) REFERENCES images(id)
);