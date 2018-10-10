import pyautogui as pya

width, height = pya.size();
print(width); print(height);
"""
for i in range(10):
    pya.moveTo(100, 100, duration=0.25)
    pya.moveTo(200, 100, duration=0.25)
    pya.moveTo(200, 200, duration=0.25)
    pya.moveTo(100, 200, duration=0.25)
"""