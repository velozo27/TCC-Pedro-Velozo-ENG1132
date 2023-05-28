from pynput.mouse import Controller, Button
import time
mouse = Controller()

while True:
    # Move pointer relative to current position
    mouse.move(5, -5)
    time.sleep(1)
    mouse.move(-5, 5)

    mouse.click(Button.left, 1)
    # mouse.move(5, 5)
    
    # mouse.move(-5, 5)
    print('clicked')

    time.sleep(59)
