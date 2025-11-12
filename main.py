#!/usr/bin/env python

import game_thread
import sys


def main():
    play_mode = any(arg in ("--play", "-p") for arg in sys.argv[1:])
    game = game_thread.GameThread(1)
    if play_mode:
        # Human vs AI using latest model
        game.loop_human_vs_ai()
    else:
        # Training: background thread generates data and trains; loop draws board
        game.start()
        game.loop()


if __name__ == "__main__":
    main()





