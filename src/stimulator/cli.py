"""Command-line interface."""

import typer
from dotenv import load_dotenv

import stimulator.discord

app = typer.Typer()
app.add_typer(stimulator.discord.app, name="discord", help="Discord Scraper CLI")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Stimulator CLI."""
    load_dotenv()
