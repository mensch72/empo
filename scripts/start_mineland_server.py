#!/usr/bin/env python3
"""
Start MineLand Minecraft server and keep it running.

This script is used by the mineland Docker container to spawn a Minecraft
server via MineLand and keep it alive for external connections from empo-dev.
"""
import mineland
import time
import signal
import sys


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    print("Shutting down MineLand server...")
    sys.exit(0)


def main():
    """Start MineLand and keep the Minecraft server running."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("Creating MineLand environment (spawns Minecraft server)...")
    print("This will take 1-2 minutes on first run...")
    print("")

    try:
        env = mineland.make(
            task_id="playground",
            agents_count=1,
            headless=True,
        )
        print("✓ Minecraft server started!")
        print("Server accessible at mineland:25565 from empo-dev")
        print("")
        print("Keeping server alive...")

        # Keep the environment alive by taking no-op actions
        obs = env.reset()
        while True:
            action = [mineland.Action()]
            obs, code_info, event, done, task_info = env.step(action)
            time.sleep(1)

    except Exception as e:
        print(f"Error starting MineLand: {e}")
        import traceback
        traceback.print_exc()
        print("")
        print("Keeping container running for debugging...")
        print("Check the error above and restart with: docker restart mineland")
        # Keep container running for debugging
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
