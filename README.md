# lootreport
Capture data from Lords Mobile monster loots.
This script is provided as-is.
It is not user-friendly.
It will click on your screen to delete gifts.
Beware and use it at your own risk.

## Requirements
pyautogui
pyscreenshot
Pillow (the PIL fork)
pytesseract

## Installation
1) Install Google's Tesseract, e.g. for Windows:
https://github.com/UB-Mannheim/tesseract/wiki
2) Download the script.
3) With your Android emulator open, capture the screen while you are in the monster loots tab (see coords_explanations/example.png).
4) Open the script with an IDE.
5) Adjust the coordinates in the Coords class to your screen size and resolution. Use the images in coords_explanations/ as a guide.

## Usage

### Saving images for capture
1) Set your monitor to its brightest (color brightness can affect image checks). 
2) On Nox, open all loots.
3) Set the `task` variable to "screen".
4) Run the script and quickly return to Nox to start capturing.
5) When the capture is complete or stops, the cursor will move to the upper-left corner of the screen.
6) IF YOU NEED TO STOP THE SCRIPT, move the cursor in one of the four corners of the screen. The script will stop automatically.

### Capturing data
1) Set the `task` variable to "loadnew" (if starting from scratch) or "load" (if you want to append new captures to existing data).
2) Run the script; it will load all images in the "saved" directory, capture, and generate the `current_results.tsv` and `AllData.tsv` files, plus the `AllData.pickled` file. In addition, the valid images are accessible in the script through the `images` object.

### Amending data (correcting hunter names, etc)
1) Add a typo to the `hunter_typos` dictionary.
2) Set the `task` variable to "retypo".
3) Run the script.

### Checking why an image was rejected
`load_and_test("imagename")`
...you might need to adjust green/blue parameters if your screen colors are a bit different than mine.
