import input_output
import media_pipe_handler
import ai_handler
import camera_handler
import time
from threading import Thread

Thread(target=input_output.main, daemon=True).start()
Thread(target=media_pipe_handler.start_threads, daemon=True).start()

try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    media_pipe_handler.end_program()