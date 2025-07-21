[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/galacticglum/stimulator/main.svg)](https://results.pre-commit.ci/latest/github/galacticglum/stimulator/main)

# **Stimulator**

Multimodal, persona-aware conversational AI

## 🚀 Overview

Simulating realistic conversations with persona-aware dialogue LLMs

## 📦 Package Management

This project uses [Poetry](https://python-poetry.org/) to manage dependencies.

To install dependencies:

```bash
poetry install
```

For installation instructions and more details, refer to the [Poetry documentation](https://python-poetry.org/docs/).

## 🛠️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/galacticglum/stimulator.git
cd stimulator
```

### 2. Set up environment variables

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

The default values are intended for **development** mode. You should review each variable and update it as needed, especially credentials and API keys.

To load the environment variables:

```bash
source .env
```

## Discord Integration

If you're scraping data from Discord and/or using the simulator to interact with Discord channels, you'll need to set up a Discord user token. To do so, set the `DISCORD_USER_TOKEN` variable in your `.env` file using a **user token**. A user token is a personal access token that allows you to interact with Discord as a user, rather than as a bot. You can follow this [guide to obtain your token](https://gist.github.com/MarvNC/e601f3603df22f36ebd3102c501116c6).

> ⚠️ **Warning**: Using a Discord self-token (i.e., user token) is against Discord’s [Terms of Service](https://discord.com/terms) and may result in account suspension or termination. Proceed at your own risk.