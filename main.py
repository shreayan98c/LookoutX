from LookoutX.utils import read_config
from LookoutX.dataset import stream
import click
from rich.logging import RichHandler
import logging


@click.group()
def cli():
    logging.basicConfig(
        level="DEBUG",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@cli.command()
# @click.option("--batch-size", default=32)
# @click.option("--epochs", default=10)
# @click.option("--lr", default=1e-3)
# @click.option("--log-interval", default=100, help="Number of batches between logging")
def train():

    # read config file and get the wifi ip and port
    config = read_config(path='config.ini', section='Livestream')

    # extract the video and audio streams from the url
    stream(wifi_ip=config['wifi_ip'], port=config['port'])


if __name__ == "__main__":
    # cli()
    train()
