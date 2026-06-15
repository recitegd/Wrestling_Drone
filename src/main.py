import input_output
import media_pipe_handler
import ai_handler
import wrestler_tracker
import time
from threading import Thread

Thread(target=input_output.main, daemon=True).start()
Thread(target=wrestler_tracker.camera_stream_thread, daemon=True).start()

try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    wrestler_tracker.end_program()
