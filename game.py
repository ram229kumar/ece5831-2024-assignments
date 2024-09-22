#game.py
import random
import draw

def play_game():
    return random.choice(["Win", "Lose","Tie"])

def main():
    draw.draw_line()
    result = play_game()
    draw.draw_game(result)
    draw.draw_line()

if __name__ == '__main__':
    main()