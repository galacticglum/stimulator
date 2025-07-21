"""Command-line interface."""

import typer
from dotenv import load_dotenv

import stimulator.discord_scraper
import stimulator.pc_model

app = typer.Typer()
app.add_typer(
    stimulator.discord_scraper.app, name="discord-scraper", help="Discord Scraper CLI"
)
app.add_typer(stimulator.pc_model.app, name="pc-model", help="PC Model CLI")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Stimulator CLI."""
    load_dotenv()
