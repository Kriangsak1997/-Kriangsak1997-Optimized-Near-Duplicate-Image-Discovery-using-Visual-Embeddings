import datetime
import os
import sys
import numpy as np
import logging
import glob
from connector import connect

Log_Format = "%(asctime)s - %(message)s"


def crawl(path: str) -> None:
    pid = os.getpid()
    logging.basicConfig(filename=f"logs/1_1_1_1/benchmark_{pid}_crawled.log",
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

    # bench = [filename for filename in glob.iglob(path + '**/*.jpg', recursive=True)]
    # bench = np.sort(np.array(bench))
    # # print(len(bench))
    # labels = [filename.split("/")[7] for filename in bench]
    # benchmark_data = [ labels[i]+ "-" + str(i) + "-" + bench[i] for i in range(len(bench))]

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
        # test = txt_file_reader.reader("paths.txt")[0]
        # inp = txt_file_reader.reader("seniorProject/image/INRIA/")
        large = "/Users/kriangsakthuiprakhon/Documents/seniorProject/image/101_ObjectCategories/"
        crawl(large)
        # train = "/Users/kriangsakthuiprakhon/Documents/seniorProject/image/INRIA"
        # crawl(train)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
