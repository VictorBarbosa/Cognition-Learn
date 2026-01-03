import uuid
from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage, OutgoingMessage

class VisualMonitorSideChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("534c891e-810f-11ef-8a6a-325096b39f47"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        # We don't expect messages back from Unity in this scenario
        pass

    def send_champion_info(self, algorithm: str, port: int, step: int, checkpoint_path: str) -> None:
        """
        Sends the Champion model info to the Unity Visual Monitor.
        Protocol:
        1. Algorithm Name (String)
        2. Port (Int32)
        3. Step (Int32)
        4. Checkpoint Path (String)
        """
        msg = OutgoingMessage()
        msg.write_string(algorithm)
        msg.write_int32(port)
        msg.write_int32(step)
        msg.write_string(checkpoint_path)
        super().queue_message_to_send(msg)
