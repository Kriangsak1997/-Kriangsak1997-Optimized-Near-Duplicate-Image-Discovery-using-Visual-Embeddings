# reading file
import logging
import string
import typing
from typing import List


def reader(paths: string) -> List[str]:
    # data
    try:
        with open(paths, 'r') as file:
            lines = [line.rstrip('\n') for line in file]
            return lines
    except Exception as e:
        logging.error('Error at %s', 'division', exc_info=e)


# print(reader("seniorProject/image/INRIA/"))
