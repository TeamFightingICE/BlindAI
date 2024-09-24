from src.trained_ai.fight_agent import SoundAgent
from loguru import logger
from pyftg.socket.aio.gateway import Gateway
import os 

async def run_fight(encoder: str, p2: str, game_num: int):
    character = "ZEN"
    host = os.environ.get("SERVER_HOST", "127.0.0.1")
    port = 31415
    gateway = Gateway(host, port)
    ai_name = "BlindAI"
    blind_ai = SoundAgent(logger=logger, encoder=encoder, path="trained_model", rnn=True)
    gateway.register_ai(ai_name, blind_ai)
    await gateway.run_game([character, character], [ai_name, p2], game_num)