# Picarch

Picarch is a Python project for face detection and image similarity search using insightface and PostgreSQL. The project detects faces in images, encodes them, stores the embeddings along with image paths in a PostgreSQL database, and allows searching for similar images.

## Inspiration

I had a collection of 12k+ photos and was too lazy go through all the photos and find pictures with me, so I built this project.

I overengineered a problem and I'm opensourcing it so you don't have to - clone it and save some time in you life 🙃

## Features

- Face detection and embedding using InsightFace.
- Image storage and similarity search using PostgreSQL with vector data.
- Command line interface to encode images, search for similar faces, and manage the database.

## Setup

### Prerequisites

- Python 3.12+
- PostgreSQL database server

### Installation

1. Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/SirusCodes/picarch.git
cd picarch
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirement.txt
```

4. Setup the database:

Setup [pgvector in Postgres](https://medium.com/@besttechreads/step-by-step-guide-to-installing-pgvector-and-loading-data-in-postgresql-f2cffb5dec43) or you can use a [docker image](https://hub.docker.com/r/pgvector/pgvector).

## Running the Project

Picarch provides several command line commands:

### 1. Encode Images

Recursively search a directory for images, encode faces, and store the embeddings in the database.

```bash
python main.py encode <path_to_images>
```

### 2. Search for a Person

Provide an image of a face to search for similar images in the database.

```bash
python main.py search <image_path> [--output <output_directory>]
```

### 3. Truncate Tables

Truncate all image and embedding tables in the database.

```bash
python main.py truncate
```

### 4. Drop Tables

Drop the tables from the database.

```bash
python main.py drop
```

## Project Structure

- `src/`
  - `ml.py`: Contains the image encoding functions.
  - `db/`: Contains database utils and classes to handle PostgreSQL operations.
- `database.ini`: Configuration file for the PostgreSQL connection.
- `requirement.txt`: Lists the project dependencies.
- `main.py`: CLI entry point for the project.
