import pika
from dotenv import dotenv_values

config = dotenv_values("foo.bar")


def connect() -> pika.connection:
    credentials = pika.PlainCredentials(username=config['USER'], password=config['PASSWORD'])
    parameters = pika.ConnectionParameters(credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    return connection
