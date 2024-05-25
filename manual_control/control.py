import getch

while True:
    key = getch.getch()
    if key == 'f': break
    with open('command.txt', 'w') as file:
        file.write(key)

