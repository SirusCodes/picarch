import concurrent.futures
from multiprocessing import Queue
import os
import shutil
import threading
import time
import logging
from rich.progress import Progress, TimeElapsedColumn, MofNCompleteColumn
import argparse

import src.ml as ml
import src.db.db as db

logger = logging.getLogger('picarch')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

progress = Progress(
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    MofNCompleteColumn(),
)

def encode_image(image_uri: str):
    try:
        embeddings = ml.encode(image_uri)
        return embeddings
    except Exception as e:
        logger.error(f"Process: Error embedding {image_uri}: {e}")

def update_db(picarch: db.Picarch, queue: Queue):
    logger.debug("Thread: Database thread started")
    try:
        while True:
            if not queue.empty():
                image_uri, embeddings = queue.get()
                if image_uri is None:
                    logger.info("Thread: Database thread received None, exiting")
                    break
                if embeddings is None:
                    continue
                image_id = picarch.insert_image_path(image_uri)
                picarch.insert_image_embeddings(image_id, embeddings)
            else:
                time.sleep(1)
    except Exception as e:
        logger.error(f"Thread: Database thread error: {e}")

def get_images_from_path(path):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(image_extensions):
                images.append(os.path.join(root, file))
    logger.info(f"Main: Found {len(images)} image(s).")
    return images

def run_truncate(picarch: db.Picarch):
    logger.info("Truncating the database...")
    picarch.trunate_tables()
    logger.info("Database truncated.")

def run_drop(picarch: db.Picarch):
    logger.info("Dropping the database...")
    picarch.drop_tables()
    logger.info("Database dropped.")

def main():
    parser = argparse.ArgumentParser(description="Picarch CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    encode_parser = subparsers.add_parser("encode", help="Encode images from a given directory")
    encode_parser.add_argument("path", help="Path to search for images")

    search = subparsers.add_parser("search", help="Search for person in images")
    search.add_argument("path", help="Person to search for")
    search.add_argument("--output", help="Output folder for found images", default="find")

    subparsers.add_parser("truncate", help="Truncate the tables")

    subparsers.add_parser("drop", help="Drop the tables")


    args = parser.parse_args()

    picarch = db.Picarch()

    if args.command == "encode":
        run_processes(args.path, picarch)
    elif args.command == "truncate":
        run_truncate(picarch)
    elif args.command == "drop":
        run_drop(picarch)
    elif args.command == "search":
        run_search(args.output, args.path, picarch)
    else:
        parser.print_help()

def run_search(output: str,path: str , picarch: db.Picarch):
    logger.info("Searching for person...")
    embeddings = ml.encode(path)
    if len(embeddings) != 1:
        logger.error("Please provide exactly one face to search for.")
        return
    results = picarch.get_image_similar_to_embedding(embeddings[0])
    logger.info(f"Found {len(results)} result(s).")
    find_folder = os.path.join(os.getcwd(), output)
    os.makedirs(find_folder, exist_ok=True)
    for result in results:
        destination = os.path.join(find_folder, os.path.basename(result[0]))
        try:
            if os.path.exists(result[0]):
                shutil.copy2(result[0], destination)
        except Exception as e:
            logger.error(f"Error copying {result[0]} to {destination}: {e}")
    logger.info("Done searching.")

def get_images_not_embedded(picarch: db.Picarch, images: list[str]):
    logger.info("Main: Checking for images that need embedding...")
    embedded_images = picarch.get_all_image_paths()
    embedded_images = [image[0] for image in embedded_images]
    logger.info(f"Main: Found {len(embedded_images)} embedded image(s).")

    job_remaining = []
    for image in images:
        if image not in embedded_images:
            job_remaining.append(image)
    logger.info(f"Main: Found {len(job_remaining)} image(s) to embed.")
    return job_remaining

def run_processes(path, picarch):
    images = get_images_from_path(path)

    images = get_images_not_embedded(picarch, images)

    queue = Queue()
    db_thread = threading.Thread(target=update_db, args=(picarch, queue,))
    db_thread.start()

    logger.info("Main: embedding images...")
    with progress:
        task = progress.add_task("[cyan]Embedding images...", total=len(images))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for uri, embedding in zip(images, executor.map(encode_image, images)):
                progress.update(task, advance=1)
                if embedding is None or len(embedding) == 0:
                    progress.console.log(f"[bold red]Main: No face found in {uri}[/bold red]")
                    continue
                progress.console.log(f"Embedded {uri}")
                queue.put((uri, embedding))
    logger.info("Main: Done embedding images.")

    queue.put((None, None))
    db_thread.join()

if __name__ == "__main__":
    main()