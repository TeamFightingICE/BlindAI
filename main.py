import asyncio

import typer
from loguru import logger
from typing_extensions import Annotated, Optional

from src.analyze import analyze_fight_result
from src.core import async_train_process
from src.enums.encoder import EncoderEnum
from src.enums.opponent_ai import OpponentAIEnum
from src.visualize import draw_polynomial

app = typer.Typer()


@app.command()
def train(
        id: Annotated[str, typer.Option(help="Experiment id")],
        encoder: Annotated[EncoderEnum, typer.Option(help="Encoder type")],
        p2: Annotated[OpponentAIEnum, typer.Option(help="The opponent AI")],
        recurrent: Annotated[bool, typer.Option(help="Use GRU")] = False,
        n_frame: Annotated[int, typer.Option(help="Number of frame to sample data")] = 1,
        epoch: Annotated[int, typer.Option(help="Number of epochs to train")] = 10,
        training_iteration: Annotated[int, typer.Option(help="Number of training iterations")] = 60,
        game_num: Annotated[int, typer.Option(help="Number of games to play per iteration")] = 5,
        port: Annotated[Optional[int], typer.Option(help="Port used by DareFightingICE")] = 31415):
    input_params = {
        'id': id,
        'encoder': encoder.value,
        'p2': p2.value,
        'recurrent': recurrent,
        'n_frame': n_frame,
        'epoch': epoch,
        'training_iteration': training_iteration,
        'game_num': game_num,
        'port': port
    }
    logger.info('Input parameters:')
    logger.info(input_params)
    asyncio.run(async_train_process(encoder.value, id, p2.value, recurrent, n_frame, epoch, training_iteration, game_num, port))


@app.command()
def analyze(
        path: Annotated[str, typer.Option(help="The directory containing result log")]):
    win_ratio, hp_diff = analyze_fight_result(path)
    print('The winning ratio is:', win_ratio)
    print('The average HP difference is:', hp_diff)


@app.command()
def visualize(
        file: Annotated[str, typer.Option(help="The result file")],
        title: Annotated[str, typer.Option(help="Title of the plot")],
        degree: Annotated[int, typer.Option(help="Polynomial degree")] = 4):
    print('Input parameters:')
    print(f'file={file} title={title} degree={degree}')
    draw_polynomial(file, title, degree)


if __name__ == '__main__':
    app()
