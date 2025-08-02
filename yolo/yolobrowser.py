import sys
import time
import requests
from io import BytesIO
import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from ultralytics import YOLO
import cv2
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException

MODEL_PATH = r'/runs/detect/train/weights/best.pt'
TARGET_HEIGHT = 200
TARGET_WIDTH = 300

try:
    print(f"Loading YOLO model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    sys.exit(f"Failed to load YOLO model: {e}")

def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def detect_slider_position(base_image):
    time.sleep(1)
    img = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
    
    results = model(img)
    
    for result in results:
        boxes = result.boxes
        
        if len(boxes) == 0:
            print("No slider cutout detected.")
            return None
        box = boxes[0]
        
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        print(f"Detected slider cutout at ({center_x:.2f}, {center_y:.2f})")
        
        return center_x
    
    return None

def detect_captcha_image(driver):
    background_element = driver.find_element(By.CLASS_NAME, 'geetest_bg')
    background_style = background_element.get_attribute('style')
    background_url = background_style.split('url("')[1].split('");')[0]
    return background_url

def init_browser():
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-notifications')
    options.add_argument('--start-maximized')
    driver = webdriver.Chrome(options=options)
    return driver

def drag_slider(driver, distance):
    time.sleep(1)
    
    try:
        slider = driver.find_element(By.XPATH, '//*[@id="captcha"]/div[2]/div[1]/div[4]/div[1]/div[2]/div/div/div[2]/div/div[3]')
    except Exception as e:
        print("Failed to locate the slider element:", e)
        return
    
    def get_track(distance):
        track = []
        steps = 10
        acceleration_steps = 6
        deceleration_steps = 4

        for i in range(acceleration_steps):
            move = (distance / (acceleration_steps + deceleration_steps)) * (i + 1) / acceleration_steps
            track.append(round(move))

        constant_steps = steps - acceleration_steps - deceleration_steps
        for _ in range(constant_steps):
            move = distance / steps
            track.append(round(move))

        for i in range(deceleration_steps):
            move = (distance / (acceleration_steps + deceleration_steps)) * (deceleration_steps - i) / deceleration_steps
            track.append(round(move))

        total_moved = sum(track)
        remaining = distance - total_moved
        if remaining > 0:
            track[-1] += round(remaining)

        return track


    
    track = get_track(distance)
    
    action = ActionChains(driver)
    action.click_and_hold(slider).perform()
    for x in track:
        action.move_by_offset(xoffset=x, yoffset=0).perform()
    action.release().perform()
    
    print("Slider moved.")

def wait_for_page_load(driver, timeout=30):
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        print("Page loaded completely.")
    except TimeoutException:
        print(f"Page did not load within {timeout} seconds.")
        driver.quit()
        sys.exit(1)

def wait_for_element_visibility(driver, by, identifier, timeout=30):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((by, identifier))
        )
        print(f"Element '{identifier}' is visible.")
        return element
    except TimeoutException:
        print(f"Element '{identifier}' not visible after {timeout} seconds.")
        driver.quit()
        sys.exit(1)

def wait_and_click(driver, by, identifier, timeout=30):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((by, identifier))
        )
        element.click()
        print(f"Clicked on element '{identifier}'.")
    except TimeoutException:
        print(f"Element '{identifier}' not clickable after {timeout} seconds.")
        driver.quit()
        sys.exit(1)
    except ElementClickInterceptedException:
        print(f"Click intercepted for element '{identifier}'. Attempting JavaScript click.")
        try:
            driver.execute_script("arguments[0].click();", element)
            print(f"JavaScript click executed for element '{identifier}'.")
        except Exception as e:
            print(f"JavaScript click failed for element '{identifier}': {e}")
            driver.quit()
            sys.exit(1)

def close_overlays(driver):
    try:
        close_buttons = driver.find_elements(By.CLASS_NAME, 'modal-close-button')
        for btn in close_buttons:
            btn.click()
            print("Closed an overlay/modal.")
            time.sleep(1)
    except Exception as e:
        print(f"No overlays to close or failed to close overlays: {e}")


def main():
    driver = init_browser()
    driver.set_window_size(1920, 1080)
    driver.get('https://www.geetest.com/en/adaptive-captcha-demo') 
    wait_for_page_load(driver, timeout=30)
    slider_xpath = '/html/body/div[1]/div/div[13]/div/section/div/div[2]/div[1]/div[2]/div[3]/div[3]'
    close_overlays(driver)
    
    wait_and_click(driver, By.XPATH, slider_xpath, timeout=15)
    clicktoverify = '/html/body/div[1]/div/div[13]/div/section/div/div[2]/div[2]/div[2]/form/div[3]/div[2]/div[1]/div[1]'
    wait_and_click(driver, By.XPATH, clicktoverify, timeout=15)

    try:
        captcha_bg = wait_for_element_visibility(driver, By.CLASS_NAME, 'geetest_bg', timeout=15)
        print("CAPTCHA appeared.")
    except Exception as e:
        print("CAPTCHA did not appear:", e)
        driver.quit()
        return
    
    time.sleep(2)
    
    try:
        background_image_url = detect_captcha_image(driver)
        print(f"Background image URL: {background_image_url}")
    except Exception as e:
        print("Failed to detect CAPTCHA background image:", e)
        driver.quit()
        return
    
    try:
        background_image = download_image(background_image_url)
        background_image.save("background_image.png") 
    except Exception as e:
        print("Failed to download background image:", e)
        driver.quit()
        return
    
    try:
        detected_position = detect_slider_position(background_image)
        if detected_position is None:
            print("Could not detect slider position.")
            driver.quit()
            return
        print(f"Detected slider position: {detected_position:.2f} pixels")
    except Exception as e:
        print("Failed to detect slider position:", e)
        driver.quit()
        return
    
    detected_position = max(0, min(detected_position, TARGET_WIDTH))
    print(f"Adjusted detected slider position: {detected_position:.2f} pixels")
    
    distance = detected_position - 40  
    
    print(f"Calculated movement distance: {distance:.2f} pixels")
    drag_slider(driver, distance)
    time.sleep(5)
    driver.quit()

if __name__ == "__main__":
    main()
