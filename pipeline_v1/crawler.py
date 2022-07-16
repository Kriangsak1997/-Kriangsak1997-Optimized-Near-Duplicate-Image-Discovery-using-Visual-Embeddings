import os
import sys
import numpy as np
import logging
import glob
from connector import connect
from dotenv import dotenv_values
ls

config = dotenv_values("foo.bar")
dir = config["log_dir"]
name = config["name"]
Log_Format = "%(asctime)s - %(message)s"


def crawl(path: str) -> None:
    pid = os.getpid()
    logging.basicConfig(filename=f"{dir}{name}_{pid}_crawled.log",
                        filemode="w",
                        format=Log_Format,
                        level=logging.INFO)

    logger = logging.getLogger()
    logger.info(f"PID {pid} started Crawling")
    connection = connect()
    channel = connection.channel()
    channel.queue_declare(queue='path')

    filenames = [filename for filename in glob.iglob(path + '**/*.jpg', recursive=True)]
    filenames = np.sort(np.array(filenames))
    labels = [filename.split("/")[-1][1:4] for filename in filenames]
    paths = [labels[i] + "-" + str(i) + "-" + filenames[i] for i in range(len(filenames))]

    for item in paths:
        channel.basic_publish(exchange='',
                              routing_key='path',
                              body=item,
                              mandatory=True)
    for i in range(1):
        channel.basic_publish(exchange='',
                              routing_key='path',
                              body="0-0-0",
                              mandatory=True)
    connection.close()
    print("Done Crawling")
    logger.info(f"PID {pid} done Crawling")


if __name__ == '__main__':
    try:
        dataset = config["data_set"]
        crawl(dataset)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
