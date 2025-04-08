import psycopg2
from pgvector.psycopg2 import register_vector
from . import utils

class Picarch:
    def __init__(self):
        self.conn = utils.connect(utils.load_config())
        self.create_tables(self.conn)

    def insert_image_path(self, path: str) -> int:
        """ insert a new image path into the image_paths table """
        sql = """INSERT INTO image_paths(path) VALUES(%s) RETURNING id;"""
        with self.conn.cursor() as cur:
            cur.execute(sql, (path,))
            image_id = cur.fetchone()[0]
            self.conn.commit()
            return image_id

    def get_all_image_paths(self) -> list:
        """ get all image paths from the image_paths table """
        sql = """SELECT path FROM image_paths;"""
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()


    def get_image_similar_to_embedding(self, embedding: list) -> list:
        """ get similar images to the given image embedding """
        sql = """
        SELECT DISTINCT ON (ip.path)
        ip.path, ie.embeddings <=> %s::vector as distance
        FROM image_embeddings ie
        JOIN image_paths ip ON ie.image_id = ip.id
        WHERE 1 - (ie.embeddings <=> %s::vector) >= 0.4
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (embedding, embedding,))
            return cur.fetchall()

    def insert_image_embeddings(self, image_id: int, embeddings: list[list]):
        """ insert multiple image embeddings into the image_embeddings table """
        for embedding in embeddings:
            if len(embedding) != 512:
                raise ValueError("Each embedding must have exactly 512 elements.", image_id)
        sql = """INSERT INTO image_embeddings(image_id, embeddings) VALUES(%s, %s);"""
        with self.conn.cursor() as cur:
            for embedding in embeddings:
                cur.execute(sql, (image_id, embedding))
                self.conn.commit()


    def create_tables(self, conn: psycopg2.extensions.connection):
        """ create tables in the PostgreSQL database """
        commands = (
            """
            CREATE EXTENSION IF NOT EXISTS vector;
            """,
            """
            CREATE TABLE IF NOT EXISTS image_paths (
                id SERIAL PRIMARY KEY,
                path TEXT NOT NULL UNIQUE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS image_embeddings (
                id SERIAL PRIMARY KEY,
                image_id INTEGER NOT NULL,
                embeddings VECTOR(512) NOT NULL,
                FOREIGN KEY (image_id) REFERENCES image_paths (id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS image_clusters (
                id SERIAL PRIMARY KEY,
                image_id INTEGER NOT NULL,
                cluster INTEGER NOT NULL,
                FOREIGN KEY (image_id) REFERENCES image_paths (id) ON DELETE CASCADE
            )
            """
        )

        try:
            # create a cursor
            with conn.cursor() as cur:
                # create table one by one
                for command in commands:
                    cur.execute(command)
                # close communication with the PostgreSQL database server
                cur.close()
                # commit the changes
                conn.commit()
                register_vector(conn)
                print('Tables created successfully.')
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def trunate_tables(self):
        """ truncate tables in the PostgreSQL database """
        commands = (
            """
            TRUNCATE TABLE image_paths CASCADE;
            """,
            """
            TRUNCATE TABLE image_embeddings CASCADE;
            """,
            """
            TRUNCATE TABLE image_clusters CASCADE;
            """
        )

        try:
            # create a cursor
            with self.conn.cursor() as cur:
                # create table one by one
                for command in commands:
                    cur.execute(command)
                # close communication with the PostgreSQL database server
                cur.close()
                # commit the changes
                self.conn.commit()
                print('Tables truncated successfully.')
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def drop_tables(self):
        """ drop tables in the PostgreSQL database """
        commands = (
            """
            DROP TABLE IF EXISTS image_paths CASCADE;
            """,
            """
            DROP TABLE IF EXISTS image_embeddings CASCADE;
            """,
            """
            DROP TABLE IF EXISTS image_clusters CASCADE;
            """
        )

        try:
            # create a cursor
            with self.conn.cursor() as cur:
                # create table one by one
                for command in commands:
                    cur.execute(command)
                # close communication with the PostgreSQL database server
                cur.close()
                # commit the changes
                self.conn.commit()
                print('Tables dropped successfully.')
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def __del__(self):
        self.conn.close()