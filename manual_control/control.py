import getch

while True:
    key = getch.getch()
    if key == 'f': break
    with open('manual_control/command.txt', 'w') as file:
        file.write(key)

