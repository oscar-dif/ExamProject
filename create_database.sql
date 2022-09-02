DROP DATABASE IF EXISTS houses_db;
DROP USER IF EXISTS houses_user;

CREATE DATABASE houses_db;
CREATE USER houses_user WITH ENCRYPTED PASSWORD '123';
GRANT ALL PRIVILEGES ON DATABASE houses_db TO houses_user;

\c houses_db houses_user