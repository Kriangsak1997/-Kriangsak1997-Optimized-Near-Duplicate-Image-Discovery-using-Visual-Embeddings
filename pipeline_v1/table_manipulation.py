import os
import sys
import serializer
import schema
import logging
import LSH
from connector import connect
from dotenv import dotenv_values
config = dotenv_values("foo.bar")
dir = config["log_dir"]
name = config["name"]
Log_Format = "%(asctime)s - %(message)s"
pid = os.getpid()
logging.basicConfig(filename=f"{dir}{name}_{pid}_table_manipulation.log",
                    filemode="w",
                    format=Log_Format,
                    level=logging.INFO)

logger = logging.getLogger()

def main() -> None:
    # counter = Count()
    connection = connect()
    channel = connection.channel()
    channel.queue_declare(queue='hash')

    def callback(ch, method, properties, body: bytes) -> None:

        sc, data = serializer.dReader(body, schema.hash_schema)
        label = data[0]["label"]
        id = data[0]["id"]
        hash = data[0]["hash_value"]
        logger.info(f"image: {id}_{label} arrived at table_manipulation")
        if label == id == hash:
            # counter.count = counter.count + 1
            # if counter.count == 2:
            channel.stop_consuming()
        else:
            logger.info(f"adding: {id}_{label} to LSH")
            LSH.lsh_builder.lsh.add(id, label, hash)
            logger.info(f"added: {id}_{label} to LSH")
            # print(f"PID {pid} added {id} to table")

    channel.basic_consume(queue='hash', on_message_callback=callback, auto_ack=True)
    print("started table manipulation")
    channel.start_consuming()
    print("completed table manipulation")
    logger.info(f"completed adding to LSH")
    connection.close()


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
