import psutil
import os


def get_memory_usage():

    process = psutil.Process(
        os.getpid()
    )

    memory_mb = (
        process.memory_info().rss
        / 1024
        / 1024
    )

    return memory_mb