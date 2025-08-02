 # GeeTest/DataDome Slide Captcha Solver

## Introduction
- Solves all types of puzzle piece captchas
- Created as a proof of concept to show how front-end captchas are vulnerable to AI solving
- Also wanted to build an add-on to my previously published project [which bypasses DataDome's detection almost every time](https://github.com/Millionarc/datadome-cookie-generator)

## Methodology
 ### Captcha Creation
  - Manually labeling hundreds of thousands of images would be infeasible which requires us to create our own versions of their captchas
  - [GeeTest](https://www.geetest.com/en/adaptive-captcha-demo) uses a pool of cutouts and base images to create captchas, places them within a frame randomly and fades the cutout
  - [DataDome](https://datadome.co/) does not have a demo site, only pops up on a protected site or have a bad cookie or send an invalid POST request, uses random images with same cutout
 ### AI/ML Solving
 - Originally started with TensorFlow but switched to YOLOv11
 - Used YOLOv11n.pt with 7 epochs of training since it instantly plateaued, all args are [here](https://github.com/Millionarc/geetest-captcha-solver/blob/main/yolo/runs/detect/train/args.yaml)
 - Honestly, if you're looking for a ready-made solution instead of training your own model, an API like [CapSolver](https://dashboard.capsolver.com/passport/register?inviteCode=K4aa2y7rcpfX) can handle GeeTest pretty well

## Usage
 ### Training (Optional as I Added the Weights [Here](https://github.com/Millionarc/geetest-captcha-solver/blob/main/yolo/runs/detect/train/weights/best.pt))
  - Generate captchas with [generatecaptchas.py](https://github.com/Millionarc/geetest-captcha-solver/blob/main/yolo/generatecaptchas.py) using images in the assets folder
  - Create the dataset using [this script](https://github.com/Millionarc/geetest-captcha-solver/blob/main/yolo/yolomakedataset.py)
  - If you've never used YOLO before here's their [setup page](https://docs.ultralytics.com/models/yolo11/) which goes over training
 ### Selenium
  - I created a [template/POC](https://github.com/Millionarc/geetest-captcha-solver/blob/main/yolo/yolobrowser.py) you can use for solving in browser with the GeeTest demo page. 
  - You will need to change some of the captcha identifiers depending on the site
