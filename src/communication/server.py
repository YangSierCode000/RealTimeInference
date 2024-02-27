"""
This script is used to define a Server class for a Flask server.
"""
import base64
import pickle

from flask import Flask, request, jsonify, abort
from queue import Queue
from threading import Thread
from typing import Callable, Any, Optional
from time import sleep


class SplitServer:
    """
    This class defines a Server that can receive, process, and send data.
    """

    def __init__(self, in_queue: Queue, out_queue: Queue) -> None:
        """
        Initializer for the Server class.
        """
        self.app = Flask(__name__)
        self.app.add_url_rule('/intermediate', 'intermediate', self.receive_data, methods=['POST'])
        self.api_key = "secret_api_key"  # In reality, this should be securely stored and not hard-coded
        self.in_queue = in_queue
        self.out_queue = out_queue

    def check_auth(self) -> bool:
        """
        Checks if the API key is present and correct.
        """
        return request.headers.get('X-API-Key') == self.api_key

    def receive_data(self) -> Any:
        """
        Receives data and pushes it into the data_queue.
        """
        if not self.check_auth():
            abort(401)  # Unauthorized
        serialized_data = request.data
        data = pickle.loads(serialized_data)
        print(f"Received data.")

        self.in_queue.put(data)
        while self.out_queue.empty():
            sleep(0.1)
        data = self.out_queue.get()
        serialized_data = pickle.dumps(data)

        print(f"Sending intermediate data back.")
        return serialized_data

    def run(self, host: str, port: int) -> None:
        """
        Runs the Flask server.
        """
        self.app.run(host=host, port=port)


if __name__ == '__main__':
    in_queue = Queue()
    out_queue = Queue()

    server = SplitServer()
    server.run("localhost", 10086)
