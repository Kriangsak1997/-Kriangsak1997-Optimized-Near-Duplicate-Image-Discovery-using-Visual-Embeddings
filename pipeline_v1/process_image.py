import os
import time
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
import serializer
import schema
from load_process import load_process
import logging
from connector import connect

Log_Format = "%(asctime)s - %(message)s"


def main() -> None:
    pid = os.getpid()
    print(f"PID: {pid} is processing")
    logging.basicConfig(filename=f"logs/1_1_1_1/benchmark_{pid}_process_image.log",
                        filemode="w",
                        format=Log_Format,
                        level=logging.INFO)

    logger = logging.getLogger()
    logger.info(f"PID {pid} started loading and processing")
    connection = connect()
    channel = connection.channel()
    channel.queue_declare(queue='path')

    def publish(data):
        try:
            channel.queue_declare(queue="image_array")
            channel.basic_publish(exchange='',
                                  routing_key='image_array',
                                  body=data,
                                  mandatory=True)
            # print(f"PID: {pid} added to Image_array Queue ")
        except Exception as e:
            logging.error('Error at %s', 'division', exc_info=e)

    def callback(ch, method, properties, body: bytes) -> None:
        label, id, path = body.decode("utf-8").split("-")
        logger.info(f"processing started: {id}_{label} ")
        if label == id == path:
            # poison_tail = serializer.dWriter(schema.schema, [{"label": label, "id": id, "image_array": path}])
            # publish(poison_tail)
            channel.stop_consuming()

        else:
            # img = image.load_img(path, target_size=(224, 224))
            xs = load_process(path)
            msg = np.array(xs.tolist()[0]).tobytes()
            logger.info(f"processing completed: {id}_{label}")
            data = serializer.dWriter(schema.schema, [{"label": label, "id": id, "image_array": msg}])
            # publish to a new Queue
            publish(data)

    # logger.info("Logging before basic consume")
    channel.basic_consume(queue='path', on_message_callback=callback, auto_ack=True)
    logger.info("Logging before start consuming")
    channel.start_consuming()
    print(f'PID{pid} done processing')
    connection.close()
    logger.info(f"PID {pid} completed loading and processing")


if __name__ == '__main__':
    try:
        main()
        # max_worker = 2
        # with ProcessPoolExecutor(max_workers=max_worker) as executor:
        #     futures = [executor.submit(main, num) for num in range(2)]
        #     # time.sleep(1)
        #     executor.shutdown()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    finally:  # this will only execute if all the above processes are done
        poison_tail = serializer.dWriter(schema.schema, [{"label": "0", "id": "0", "image_array": "0"}])
        connection = connect()
        channel = connection.channel()
        channel.queue_declare(queue='path')
        for i in range(1):
            channel.basic_publish(exchange='',
                                  routing_key='image_array',
                                  body=poison_tail,
                                  mandatory=True)
        connection.close()
