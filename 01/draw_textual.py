# draw.py

def draw_game(result):
    print("Game result : %s" % result)

def clear_screen():
    print("_"*20)

def draw_numerical(result):
    if(result == "Win"):
        print(1)
    elif(result == "Lose"):
        print(-1)
    else:
        print(0)