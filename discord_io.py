from discordwebhook import Discord

discord = Discord(url="https://discord.com/api/webhooks/1036542332604002364/WkmNcaSB5Xkoi0lYvscWtHRDAenWhilOywGSpt_HlBv5SBLpJKh884Dbxsj4nAB1FyUd")

def send(content):
    discord.post(content=content)

def trainer_send(content):
    discord.post(
        username="Trainer Worker (single)",
        content=content
    )

def shared_storage_send(content):
    discord.post(
        username="Shared Storage Worker (single)",
        content=content
    )

def replay_buffer_send(content):
    discord.post(
        username="Replay Buffer Worker (single)",
        content=content
    )

def reanalyse_send(content):
    discord.post(
        username="Reanalyse Worker (single)",
        content=content
    )

def self_play_send(id, content):
    discord.post(
        username="Self-Play Worker (id: {})".format(id),
        content=content
    )

def test_play_send(content):
    discord.post(
        username="Test Play Worker (single)",
        content=content
    )