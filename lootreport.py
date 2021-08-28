# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 17:01:38 2018

@author: laggartolm
"""
# http://pyautogui.readthedocs.io/en/latest/cheatsheet.html

import pyautogui
import pyscreenshot as ImageGrab
import PIL
import time
import os
import pytesseract
import pandas as pd
import pickle
from pprint import pprint
import re

pyautogui.PAUSE = 0.2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

#pts.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def failsafe(text_function):
    # Decorator, return "undefined" if text extraction fails
    def wrapper(*args, **kwargs):
        try:
            text = text_function(*args, **kwargs)
        except Exception:
            text = 'undefined'
        finally:
            return text
    return wrapper


def sanitize_for_saving(text):
    return ''.join([x if (x.isalnum() or x in " -_.") else "_" for x in text])
    

class Coords:
    data_box = ((530, 300), (1150, 400)) # coordinates of the first loot (to be cropped and captured)
    green_check_pixel = (462, 11) # this pixel is only green due to the green checkmark of an open loot
    gift_check_pixel = (10, 10) # This pixel isn't deep blue when there's a loot
    block_check_pixelrange = list([(405, y) for y in range(10, 96, 10)] + # right-side column to see if there's screen messages
                                   [(8, 32)] + # left-side point between open bracket and loot image, to see if the position is wrong
                                   [(69, 84), (69, 53)] # loading circle animation
                                   )
    monster_box = (5, 5, 400, 32) # Monster info data is here
    hunter_box = (70, 38, 400, 62) # Hunter info data is here
    loot_box = (70, 70, 400, 95) # Loot info data is here
    keys_box = (545, 70, 615, 95) # Keys info data is here
    open_delete_button_pos = (1100, 330) # Click here to open or delete a loot
    rest_pos = (100, 100) # Resting cursor position


huntall = {'R5': [
         'RandomImaginaryLeaderName',
         ],
         'R4': [
         'randomimaginaryr4',
         ],
         'R3': [
         'randomimaginaryr3',
         ],
         'R2': [
         'randomimaginaryr2',
         ],
         'R1': [
         'randomimaginaryr1',
         ]
         }



hunters = [y for x in huntall.values() for y in x]

hunter_typos = {'———': '',
                }


loot_typos = (("Ghest", "Chest"),
              ("Gommon", "Common"),
              ("Glaw", "Claw"),
              ("Goins", "Coins"),
              ('Dragonaglass', 'Dragonglass'),
              ('Grusty', 'Crusty'),
              ('Grest', 'Crest'),
              ('Gore', 'Core'),
              ('Tron', 'Iron'),
              ('(4h)', '(4 h)'))


monsters = {"Titan": "Tidal Titan",
                "Maggot": "Mega Maggot",
                "Reaper": "Grim Reaper",
                "Bee": "Queen Bee",
                "Beast": "Snow Beast",
                "Drider": "Hell Drider",
                "Shaman": "Voodoo Shaman",
                "Trojan": "Mecha Trojan",
                "Appeti": "Bon Appeti",
                "Wyrm": "Jade Wyrm",
                "Summon": "Summon Bonus",
                "Slayer": "Slayer Loot",
                "Alliance": "Alliance Chest",
                "Cup": "Guild Cup Chest",
                "War": "Guild War Chest",
                "Golden": "Golden Box",
                "White": "White bonus",
                "Gottageroar": "Cottageroar"}


def hunters_report(hunt=huntall):
    tot = 0
    for r, hunters in hunt.items():
        print("================")
        tot += len(hunters)
        print(r, len(hunters))
        for hunter in hunters:
            print(hunter)
    print("Total:", tot)
    

def mark_screen_areas(image, color=(255, 255, 255)):
    # Color specific areas listed in Coords to check where they are
    tmpimg = image.copy()
    # All these areas refer to the screen, so need to use a pic of the screen
    for twotuplebox in [Coords.data_box]:
        draw_box_twotuples(tmpimg, twotuplebox, (255, 255, 255))
    for pixel in [Coords.open_delete_button_pos]:
        tmpimg.putpixel(pixel, color)
    return tmpimg


def mark_ocr_areas(raw_image, coords, color=(255, 255, 0)):
    # Color specific areas listed in Coords to check where they are
    tmpimg = raw_image.copy()
    # All these areas refer to the captured image
    for box in [coords.monster_box,
                coords.hunter_box,
                coords.loot_box,
                coords.keys_box]:
        draw_box(tmpimg, box, color)
    for pixel in [coords.green_check_pixel,
                  coords.gift_check_pixel,
                  *coords.block_check_pixelrange]:
        tmpimg.putpixel(pixel, color)
    return tmpimg


def horizontal_line(startx, endx, y):
    return [(x, y) for x in range(startx, endx+1)]

def vertical_line(x, starty, endy):
    return [(x, y) for y in range(starty, endy+1)]
    
def draw_box_twotuples(image, twotuples, color):
    draw_box(image, (twotuples[0][0], twotuples[0][1], twotuples[1][0], twotuples[1][1]), color)

def draw_box(image, box, color):
    startx, starty, endx, endy = box
    for pixel in horizontal_line(startx, endx, starty):
        image.putpixel(pixel, color)
    for pixel in horizontal_line(startx, endx, endy):
        image.putpixel(pixel, color)
    for pixel in vertical_line(startx, starty, endy):
        image.putpixel(pixel, color)
    for pixel in vertical_line(endx, starty, endy):
        image.putpixel(pixel, color)


class Image:
    def __init__(self, image, filepath, coords, imgid):
        self.filepath = filepath
        self.id = imgid
        self.coords = coords
        self.magenta = None
        self.detect(image)

        
    def detect(self, image=None):
        if image is None and self.magenta is None:
            image = PIL.Image.open(self.filepath)
        if self.magenta is None:
            self.magenta = image.convert("CMYK").getchannel(1)
        self.hunter = self.get_hunter()
        self.monster = self.get_monster()
        self.loot = self.get_loot()
        self.level = self.get_level()
        self.magenta = None
        
    def ocr(self, box):
        return pytesseract.image_to_string(self.magenta.crop(box),
                                           config=r'--psm 7').strip()

    @failsafe
    def get_monster(self):
        text = self.ocr(self.coords.monster_box).split(" ")
        text = text[-2]
        return monsters.get(text, text)
    
    @failsafe    
    def get_hunter(self):
        text = self.ocr(self.coords.hunter_box)
        return hunter_typos.get(text[10:], text[10:])

    @failsafe    
    def get_loot(self):
        text = self.ocr(self.coords.loot_box)
        for typo, correct in loot_typos:
            text = text.replace(typo, correct)
        self.goods, self.total_amount = parse_loot(text)
        return text

    @failsafe        
    def get_level(self):
        text = self.ocr(self.coords.keys_box)
        keys = int(text)
        # Transforming number of keys to monster rarity
        #summon bonuses, slayer loots etc. are 100 so count as "other"
        level = {1: 1, 2: 2, 3: 3, 5: 4, 10:5}.get(keys, "other")
        return level
    
    @failsafe
    def texts(self):
        to_text = self.monster+"_"+self.hunter+"_"+self.loot+"_"+str(self.level)
        to_text = ''.join([n for n in to_text if n in \
                ' _0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'])
        return to_text
    
    def imshow(self):
        try:
            image = PIL.Image.open(self.filepath)
            display(image)
        except (NameError, FileNotFoundError):
            print("(Unable to display image)")



def parse_loot(text):
    # Assuming all loots are <item> x quantity
    x = text.rfind('x')
    item, itemno = text[:x], text[x+1:]
    itemno = int(re.sub('[^0-9]', '', itemno))
    # Resources, gems
    if item[0] in '0123456789':
        space = item.index(' ')
        quantity = int(item[:space].replace(',', ''))
        goods = item[space+1:].strip()
        total_amount = quantity * itemno
    # Speedups, in minutes
    elif item.startswith('Speed'):
        par = item.index('(')
        goods = item[:par].strip()
        timeno, unit = re.findall('\(([\d]+)[ ]?([\w])\)', item)[0]
        total_amount = int(timeno)*{'m': 1, 'h': 60, 'd': 60*24}[unit]
    # Monster items, shields, other cases
    else:
        goods = item.strip()
        total_amount = itemno
    return goods, total_amount
       

def capture(top_left, bottom_right):
    '''
    Capture the screen grab area as image.
    The image name is given by two coordinate "tags".
    '''
    im=ImageGrab.grab(bbox=(*top_left, *bottom_right)) # X1,Y1,X2,Y2
    return im


def green(pixeldata, verbose=False):
    # Does the image have a green check mark pixel (=ok)?
    green_limits = ((15, 70), (200, 250), (35, 250))
    assert len(pixeldata) == 3
    (rlow, rhigh), (glow, ghigh), (blow, bhigh) = green_limits
    if verbose is True:
        print("Green check:", pixeldata, "[{}-{}][{}-{}][{}-{}]".format(rlow, rhigh, glow, ghigh, blow, bhigh))
    if  rlow < pixeldata[0] < rhigh:
        if glow < pixeldata[1] < ghigh:
            if blow < pixeldata[2] < bhigh:
                return True
    return False


def is_bright(pixeldata, verbose=False):
    # is the pixel all above 150? ==> used for checking if there's a gift, on a letter "e"
    assert len(pixeldata) == 3
    bright = 150
    if verbose is True:
        print("Bright check:", pixeldata, "[>{}]".format(bright))
    if all((n>150 for n in pixeldata)):
        return True
    else:
        return False

def is_not_blocked(pixeldata, verbose=False):
    # is pixeldata[0] above 45? ==> used to check if something is blocking
    assert len(pixeldata) == 3
    block = (50, 79)
    if verbose:
        print("pixeldata[0] is", pixeldata[0], "[{}-{}]".format(*block))
    if block[0] < pixeldata[0] < block[1]:
        return True
    else:
        return False


def from_dir(directory, coords=Coords, sample=None, also_save_to="processed", pastdata = None):
    # also_save_to is a directory
    if also_save_to and not os.path.isdir(also_save_to):
        os.mkdir(also_save_to)
        
    if not pastdata:
        past_ids  = []
    else:
        past_ids = [x[0] for x in pastdata]

    img_files = [x for x in os.listdir(path=directory) if x.endswith('.jpg')]
    images = []
    count = 0
    rejected = []
    for i, imgf in enumerate(img_files):
        imgid = imgf.split("_")[0]
        if imgid not in past_ids:
            print("Reading {}/{}:".format(i+1, len(img_files)), imgf)
            filepath = os.path.join(directory, imgf)
            image = PIL.Image.open(filepath)
            response = check_image(image)
            if response is True:
                im = Image(image, filepath, coords, imgid)
                images.append(im)
                if also_save_to:
                    imgname = os.path.join(also_save_to, sanitize_for_saving(\
                            "{}_{}_{}_L{}.jpg".format(imgf[:-4], im.monster, im.hunter, im.level)))
                    print("Saving:", imgname)
                    image.save(imgname)
            else:
                rejected.append(filepath)
                print(response)
            count += 1
            if sample is not None and count >= sample:
                break
    return images, rejected
    

def from_screen(directory, label='', stop = False, coords=Coords, pause_btw_clicks=1):
    global image
    if not os.path.isdir(directory):
        os.mkdir(directory)
    print("Starting in 2 seconds...")
    time.sleep(2)
    images = []
    timeone = time.time()
    saved = 0
    while True:
        print('tick')
        blocked = 0
        while True:
            image = capture(*coords.data_box)
            if all([is_not_blocked(image.getpixel(px)) for px in coords.block_check_pixelrange]):
                blocked = 0
                break
            else:
                print("Image might be blocked by some message, waiting")
                blocked += 1
                assert blocked < 10
                time.sleep(0.3)
        timeone = time.time()
        response = check_image(image)
        if response is True:
            imgname = os.path.join(directory, "{}_{}_LABEL={}.jpg".format(int(time.time()*10), "saved", label))
            print("Saving:", imgname)
            image.save(imgname)
        elif response is None:
            imgname = os.path.join(directory, "{}_notgreen.jpg".format(int(time.time()*10)))
            print("Saving:", imgname)
            image.save(imgname)
            break
        elif response is False:
            imgname = os.path.join(directory, "{}_stopping.jpg".format(int(time.time()*10)))
            print("Saving:", imgname)
            image.save(imgname)
            break
        else:
            raise AssertionError
        pyautogui.moveTo(*coords.open_delete_button_pos)
        pyautogui.click()
        saved += 1
        if stop and saved == stop:
            break
        if time.time() - timeone < pause_btw_clicks:
            time.sleep(pause_btw_clicks - (time.time() - timeone))

    pyautogui.moveTo(*coords.rest_pos)
    return images



def check_image(image, verbose=False):
    if green(image.getpixel(Coords.green_check_pixel), verbose=verbose):
        return True
    elif is_bright(image.getpixel(Coords.gift_check_pixel), verbose=verbose):
        return None
    else:
        return False
    

def tabulate(imagelist):
    data = []
    for im in imagelist:
        data.append([im.id,
                     im.monster,
                     im.level,
                     [im.hunter, 'redacted'][im.level=='other'],
                     im.loot,
                     im.goods,
                     im.total_amount])
    return data


def by_player(tabulated_data, hunters=hunters):
    score = dict([(hunter, [0, 0, 0, 0, 0, 0])for hunter in hunters])
    for imgid, monster, level, hunter, loot, goods, amount in tabulated_data:
        if type(level) != int:
            continue
        if not score.get(hunter, False):
            score[hunter] = [0, 0, 0, 0, 0, 0]
        score[hunter][level] += 1
        # Level 1 monsters are 1 point, L2=5, L3=20, L4=80, L5=350)
        score[hunter][0] += {1:1, 2:5, 3:20, 4:80, 5:350}.get(level, 0) 
    df =  pd.DataFrame.from_dict(score, orient='index', 
            columns = ['Points', 'lvl1', 'lvl2', 'lvl3', 'lvl4', 'lvl5'])
    df.index.name = 'Account'
    df = df.reindex(sorted(df.index, key=lambda x: x.lower())) #StackOverflow 30521994 u:Zero
    df["N_kills"] = df[['lvl1', 'lvl2', 'lvl3', 'lvl4', 'lvl5']].apply(sum, axis=1)
    return df[[col for col in df if col != "Points"]+["Points"]]


def show_kills_by(hunter, images, howmany=None):
    #display will only work in ipython
    num_displayed = 0
    retimg = []
    for i, image in enumerate(images):
        if image.hunter == hunter:
            retimg.append(image)
            num_displayed += 1
            print()
            print("Image number:", i)
            image.imshow()
            print("Monster: {}".format(image.monster))
            print("Hunter:  {}".format(image.hunter))
            print("Loot:    {}".format(image.loot))
            print("level:   {}".format(image.level))
        if howmany is not None and num_displayed >= howmany:
            return retimg
    return retimg


def save_processed_data(images, savefile):
    lines = ['\t'.join([str(y) for y in x]) for x in tabulate(images)]
    with open(savefile+'.tsv', 'w') as f:
        f.write('\n'.join(lines))
    # Remove OCR picture if present, to save data
    for image in images:
        image.magenta = None
    pickle.dump(images, open(savefile+'.pickled', 'wb'))

    

def load_processed_data(filename):
    print("Loading previous data...")
    # filename with no extension
    images = pickle.load(open(filename+'.pickled', 'rb'))
    with open(filename+'.tsv', 'r') as f:
        rawdata = f.read().split('\n')
    pastdata = [x.split("\t") for x in rawdata]
    # converting level to number
    for item in pastdata:
        if item[2] in '12345':
            item[2] = int(item[2])
    assert len(pastdata) == len(set([x[0] for x in pastdata]))
    print("Loaded {} kills.".format(len(pastdata)))
    return images, pastdata
    

def load_and_test(imagefile, bypassblock=False):
    img = PIL.Image.open(imagefile)
    for px in Coords.block_check_pixelrange:
        print(px, end=" ")
        print("  ==>", is_not_blocked(img.getpixel(px), verbose=True))
    block = all([is_not_blocked(img.getpixel(px)) for px in Coords.block_check_pixelrange])
    if block is True:
        print("Image can be captured")
    else:
        print("Image shouldn't be captured.")
    if any([block, bypassblock]):
        response = check_image(img, verbose=True)
        print("response:", response)
        if response is True:
            print("Image passes the green check filter, so is valid")
        elif response is False:
            print("Image doesn't pass the green check filter, but passes the 'e' "
                  "brightness filter, which means it might be an unopened gift")
        elif response is None:
            print("Image seems to be blocked.")
        else:
            print("check_image is returning an unexpected value!")
            
    return mark_ocr_areas(img, Coords)
    


if __name__ == '__main__':

    
    raw_dir = 'saved_WRONG_SAVE_BAD'
    load_dir = 'saved_EVERYTHING'
    proc_dir = 'processed_EVERYTHING'
    past_data_file = 'Data_EVERYTHING'
    resultsfile = 'results_EVERYTHING'
    
    '''
    raw_dir = 'saved'
    load_dir = 'saved'
    proc_dir = 'processed'
    past_data_file = 'AllData'
    resultsfile = 'current_results'
    '''
    
    
    
    # Tasks:
    # 'screen': capture and save images from screen
    # 'retypo': correct hunter and loot names from the current typo lists
    # 'load': load old images and OCR new images in the raw_dir folder
    # 'loadnew': reanalyze all images in the raw_dir folder
    #            ==> THIS WILL OVERWRITE pickled data and saved table!
    
    task = 'screen' # 'screen', 'retypo', 'load', 'loadnew'=> when starting a new week
    label = '2021-01-03'
    stop =  False
    delete_rejected_images = False
    
    
    if task == 'screen':
        images = from_screen(raw_dir, label, stop)
    
    if task == 'loadnew':
        check = input("Delete all data y/n? ")
        assert check == 'y'
        images = []
        tabulated_data = []
        
    elif task in ('load', 'retypo'):
        images, tabulated_data = load_processed_data(past_data_file)
    
    if task in ('load', 'loadnew'):
        new, rejected = from_dir(load_dir, also_save_to=proc_dir, pastdata=tabulated_data)
        images.extend(new) #, sample=10)
        print(len(rejected), 'images rejected:')
        for item in rejected:
            print(item)
        if delete_rejected_images:
            q = input("Delete all rejected images y/n? ")
            if q.lower() == 'y':
                for item in rejected:
                    os.remove(item)
                    print("Deleted =>", item)
                print("Done.")
        print(len(new), 'new images loaded.')

    if task == 'retypo':
        for image in images:
            new_hunter = hunter_typos.get(image.hunter, image.hunter)
            if new_hunter != image.hunter:
                print(image.id, image.hunter, "==>", new_hunter)
                image.hunter = new_hunter
            new_monster = monsters.get(image.monster, image.monster)
            if new_monster != image.monster:
                print(image.id, image.monster, "==>", new_monster)
                image.monster = new_monster
            for typo, correct in loot_typos:
                new_loot = image.loot.replace(typo, correct)
                if new_loot != image.loot:
                    print(image.id, image.loot, "==>", new_loot)
                    image.loot = new_loot
                new_goods = image.goods.replace(typo, correct)
                if new_goods != image.goods:
                    print(image.id, image.goods, "==>", new_goods)
                    image.goods = new_goods

                
    
    if task in ('load', 'loadnew', 'retypo'):
        tabulated_data = tabulate(images)
        save_processed_data(images, past_data_file)
        df = by_player(tabulated_data)
        df.to_csv(resultsfile+".tsv", sep='\t')
        #print(df)
        imghunters = set([img.hunter for img in images])
        print("These hunters are not in the list:")
        pprint(sorted(imghunters.difference(set(hunters)), key=lambda x: x.upper()))
        print("These hunters in the list made no kills so far:")
        pprint(sorted(set(hunters).difference(set(imghunters)), key=lambda x: x.upper()))
        
