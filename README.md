[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/galacticglum/stimulator/main.svg)](https://results.pre-commit.ci/latest/github/galacticglum/stimulator/main)

# **Stimulator**

Multimodal, persona-aware conversational AI for simulating Discord conversations.

## 📦 Package Management

This project uses `setuptools` for packaging and dependency management via a `setup.py` file.

To install the project and its dependencies:

```bash
pip install -e .
```

To install development dependencies:

```bash
pip install .[dev]
```

If you are running the project for the first time, you will also need to manually install a few additional dependencies:

```bash
pip install unsloth
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
pip install flash-attn --no-build-isolation
```

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

## 🤖 Discord Integration

You'll need to set up a Discord user token in your `.env` file (`DISCORD_USER_TOKEN`). This is a personal access token used to authenticate your requests as if you are a user client, not a bot. While this is not necessary for simulating conversations, it is required for scraping messages from Discord channels. Follow this [guide to obtain your token](https://gist.github.com/MarvNC/e601f3603df22f36ebd3102c501116c6).

> ⚠️ **Warning**: Using a Discord self-token (i.e., user token) is against Discord’s [Terms of Service](https://discord.com/terms) and may result in account suspension or termination. Proceed at your own risk.