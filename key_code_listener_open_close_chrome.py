from pynput import keyboard
import pyautogui
import time
import webbrowser
import os 
from webcamgpt.config import *
"""
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
svc=Service(ChromeDriverManager().install())
"""
def on_chrome_open():
    #on_chrome_close()
    print('on_chrome_open')
    webbrowser.open("http://127.0.0.1:7860")
    """
    driver = webdriver.Chrome(service=svc)
    driver.maximize_window()
    driver.get("http://127.0.0.1:7860")
    webdriver.ActionChains(driver).send_keys(Keys.F5).perform()
    webdriver.ActionChains(driver).send_keys(Keys.F11).perform()
    """

def on_chrome_close():
    print('on_chrome_close')
    pyautogui.hotkey('ctrl', 'w')
    os.system("taskkill /im chrome.exe /f")
 
listener = keyboard.GlobalHotKeys({ 
    "<f13>": on_chrome_open, 
    "<f14>": on_chrome_close
})

print("starting the keyboard listener")
listener.start()
while True:
    time.sleep(100)
listener.stop()
listener.join()
