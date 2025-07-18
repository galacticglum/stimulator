"""Command-line interface."""

import typer

import stimulator.discord_scraper as discord_scraper_cli

app = typer.Typer()
app.add_typer(discord_scraper_cli.app)
