import os
import sys
from concurrent.futures import ProcessPoolExecutor
# import time
from typing import List, Tuple
import numpy as np
import schema
import npdeserializer
import serializer
import extract
import hash_function
from connector import connect
import logging
from dotenv import dotenv_values
config = dotenv_values("foo.bar")
dir = config["log_dir"]
name = config["name"]
Log_Format = "%(asctime)s - %(message)s"


class Table:
    def __init__(self):
        self.table = dict()


def main(batch_size: int) -> None:
    pid = os.getpid()
    logging.basicConfig(filename=f"{dir}{name}_cpu_{pid}_feature_extract.log",
                        filemode="w",
                        format=Log_Format,
                        level=logging.INFO)

    logger = logging.getLogger()
    logger.info(f"PID {pid} started feature extraction")
    table = Table()
    connection = connect()
    channel = connection.channel()
    channel.queue_declare(queue='image_array')

    def publish(data: bytes) -> None:
        try:
            channel.queue_declare(queue="hash")
            channel.basic_publish(exchange='',
                                  routing_key='hash',
                                  body=data,
                                  mandatory=True)
            # print("added to Hash Queue")
        except Exception as e:
            logging.error('Error at %s', 'division', exc_info=e)

    def hash(labels: List[Tuple[str, str]], embeddings: np.ndarray) -> None:
        for (i, l), embedding in zip(labels, embeddings):
            logger.info(f"hashing: {i}_{l} started")
            val = hash_function.hash_func(embedding)
            logger.info(f"hashing: {i}_{l} completed")
            h = serializer.dWriter(schema.hash_schema,
                                   [{"label": l, "id": i, "hash_value": val}])
            publish(h)

    def feature_extract(batch_size: int) -> None:  # done here
        # read the table and dump it into ResNet
        temp = []  # dim = (x<=8,224,224,3)
        id_labels = []
        # print(f'')
        items = table.table.items()
        for item in items:
            i, la = item
            label, array = la
            id_labels.append((i, label))
            temp.append(array)
        table.table.clear()
        logger.info(f"Extracting started: {id_labels} being feature-extracted")
        embeddings = extract.extract(np.array(temp), batch_size)
        logger.info(f"Extracting completed: {id_labels} have been feature-extracted")
        hash(id_labels, embeddings)

    def callback(ch, method, properties, body: bytes) -> None:

        # print("ifCallbacked")
        s, data = serializer.dReader(body, schema.schema)
        msg = data[0]["image_array"]
        id = data[0]["id"]
        label = data[0]["label"]
        logger.info(f"image {id}_{label} arrived at feature extraction")
        if label == id == msg:
            # poison_tail = serializer.dWriter(schema.hash_schema, [{"label": label, "id": id, "hash_value": msg}])
            # print(f"PID: {pid} doing feature extraction for less than {batch_size} images")
            if len(table.table) > 0:
                # print(f'table size {len(table.table)}')
                feature_extract(batch_size=batch_size)
            # publish(poison_tail)
            channel.stop_consuming()
        else:
            # we need tensors of shape((224, 224, 3)
            array = npdeserializer.byte_deserializer(msg, (224, 224, 3))
            table.table[id] = (label, array)
        if len(table.table) == batch_size:
            # print(f"PID: {pid} doing feature extraction for {batch_size} images")
            feature_extract(batch_size=batch_size)

    channel.basic_consume(queue='image_array', on_message_callback=callback, auto_ack=True)
    print(f"PID: {pid} started CPU feature extraction")
    channel.start_consuming()  # event loops
    print(f"PID: {pid}completed CPU feature extraction")
    connection.close()
    logger.info(f"PID {pid} completed feature extraction")


if __name__ == '__main__':
    try:
        main(1)
        # max_worker = 2
        # with ProcessPoolExecutor(max_workers=max_worker) as executor:
        #     # for i in range(4):
        #     executor.map(main, [1, 1])
        #     # time.sleep(1)
        #     executor.shutdown()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    finally:  # this will only execute if all the above processes are done
        poison_tail = serializer.dWriter(schema.hash_schema,
                                                       [{"label": "0", "id": "0", "hash_value": "0"}])
        connection = connect()
        channel = connection.channel()
        channel.queue_declare(queue='hash')
        channel.basic_publish(exchange='',
                              routing_key='hash',
                              body=poison_tail,
                              mandatory=True)
        connection.close()
