from flask import Request
from werkzeug.exceptions import BadRequest


class InferSendCarRequest:
    message_id = None
    publish_time = None

    def __init__(self, req):
        data = req.get_json()
        if not data['message_id']:
            raise BadRequest(
                'InferSendCarRequest message_id not found')
        if not data['publish_time']:
            raise BadRequest(
                'InferSendCarRequest publish_time not found')
        self.message_id = data['message_id']
        self.publish_time = data['publish_time']