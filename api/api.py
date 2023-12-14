import base64
import io
import os
from PIL import Image
from pydantic import BaseModel
import cv2

import uvicorn
import sys
import torch
from fastapi import FastAPI, File, UploadFile, Request
from torchvision import models, transforms
import time
import copy
import numpy as np

sys.path.append(os.getcwd() + "/yolov5")
# from yolov5 import detect
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.augmentations import letterbox

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

model = attempt_load(os.getcwd() + "/models/best.pt", device=device)
model.eval()

th_model = attempt_load(os.getcwd() + "/models/best_th.pt", device=device)
th_model.eval()

bb_classes = {
    "gold_mine": 7,
    "elx_mine": 7,
    "dark_mine": 3,
    "th": 1,
    "eagle": 1,
    "air_def": 4,
    "inferno": 3,
    "xbow": 4,
    "wiz_tower": 5,
    "bomb_tower": 2,
    "air_sweeper": 2,
    "cannon": 7,
    "mortar": 4,
    "archer_tower": 8,
    "queen": 1,
    "king": 1,
    "warden": 1,
    "gold_storage": 4,
    "elx_storage": 4,
    "dark_storage": 1,
    "cc": 1,
    "scatter": 2,
    "champ": 1,
    "100": 1,
    "army": 1,
    "pet" : 1
}

bb_classes_arr = [
    "100",
    "air_def",
    "air_sweeper",
    "archer_tower",
    "army",
    "bomb_tower",
    "cannon",
    "cc",
    "champ",
    "dark_mine",
    "dark_storage",
    "eagle",
    "elx_mine",
    "elx_storage",
    "gold_mine",
    "gold_storage",
    "inferno",
    "king",
    "mortar",
    "pet",
    "queen",
    "scatter",
    "th",
    "warden",
    "wiz_tower",
    "xbow",
]

thnames = ["th12", "th13", "th14", "th15"]


def preprocess(image):
    transform = transforms.Compose(
        [
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
        ]
    )

    return transform(image).unsqueeze(0).to(device)


def decode_img(img):
    img_data = base64.b64decode(img)
    new_img = Image.open(io.BytesIO(img_data))

    return new_img.convert("RGB")


@app.get("/")
async def root():
    return {"status": "Healthy"}

class Detect(BaseModel):
    encoded_string: str

def pad_img(img_to_pad):
    padded_img = np.full((800, 800, 3), (114, 114, 114), dtype=np.uint8)
    padded_img = padded_img.transpose(2, 0, 1)

    _, height, width = img_to_pad.shape
    offset_width = (800 - width) // 2
    offset_height = (800 - height) // 2

    padded_img[:, offset_height:offset_height+height, offset_width:offset_width+width] = img_to_pad
    return padded_img

def reshape_copy_img(img):
    _img = letterbox(img, new_shape=(800, 800))[0]
    _img = _img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    _img = np.ascontiguousarray(_img)  # uint8 to float32
    return _img


@app.post("/burntbase")
async def detect(detect: Detect, request: Request):
    t1 = time.time()
    input_image = decode_img(detect.encoded_string)
    input_tensor = preprocess(input_image)

    image_bytes = base64.b64decode(str(detect.encoded_string))
    img_from_buf = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(img_from_buf, 1)
    img_np = img_np[:,:,:3]

    img_np = reshape_copy_img(img_np)
    img_np = pad_img(img_np)

    # asdasd
    img_to_send = torch.from_numpy(img_np).to(device)
    img_to_send = img_to_send.float()
    img_to_send /= 255.0 # 0 - 255 to 0.0 - 1.0
    if img_to_send.ndimension() == 3:
        img_to_send = img_to_send.unsqueeze(0)

    t2 = time.time()
    print("Preprocessing time: ", t2 - t1)


    with torch.no_grad():
        detection = model(input_tensor)
        th_detection = th_model(img_to_send)

    t3 = time.time()
    print("Detection time: ", t3 - t2)

    results = non_max_suppression(
        detection, conf_thres=0.7, iou_thres=0.5, classes=None, agnostic=False
    )

    th_results = non_max_suppression(
        th_detection, conf_thres=0.6, iou_thres=0.5, classes=None, agnostic=False
    )

    t4 = time.time()
    print("NMS time: ", t4 - t3)

    print("th results: ", th_results)
    townhall_level = ''
    # sort results by confidence
    sorted_results = sorted(th_results[0], key=lambda x: x[4], reverse=True)
    # print("th results: ", th_results)

    # If there is a townhall in the image, get the townhall level
    if len(sorted_results) > 0:
        # Loop through the results and get the townhall level
        # if it sees the first one, break out of the loop
        for pred in sorted_results[0]:
            x1, y1, x2, y2, conf, cls = th_results[0][0]
            print("th confidence: ", conf)
            th_bounding_box = thnames[int(cls.item())]
            if (th_bounding_box) == "th15":
                townhall_level = 15
                break
            if (th_bounding_box) == "th14":
                townhall_level = 14
                break
            if (th_bounding_box) == "th13":
                townhall_level = 13
                break

    data = []

    for pred in results[0]:
        x1, y1, x2, y2, conf, cls = pred

        try:
            class_name = bb_classes_arr[int(cls.item())]
        except:
            print("Error")
            print(cls.item())

        data.append(
            {
                "x1": x1.item(),
                "y1": y1.item(),
                "x2": x2.item(),
                "y2": y2.item(),
                "xmin": min(x1.item(), x2.item()),
                "ymin": min(y1.item(), y2.item()),
                "xmax": max(x1.item(), x2.item()),
                "ymax": max(y1.item(), y2.item()),
                "conf": conf.item(),
                "cls": class_name,
            }
        )

    # Remove duplicates if there are too many, this is a hacky way to do it
    data = max_outputs_per_class(data)

    print("Total time: ", time.time() - t1)

    return format_outputs(data, 800, 800, townhall_level)

# Remove any duplicate classes
# For example, there should only be 4 xbows, 1 eagle, 1 inferno, etc.
# If there are more than that, remove the ones with the lowest confidence
# By sorting the results by confidence, we can remove the lowest confidence ones
def max_outputs_per_class(results):
    if len(results) == 0:
        return results
    
    sorted_by_conf = sorted(results, key=lambda k: k["conf"], reverse=True)

    bb_classes_totals = copy.deepcopy(bb_classes)

    for obj in sorted_by_conf:
        if obj["cls"] in bb_classes_arr and obj["conf"]:
            if bb_classes_totals[obj["cls"]] > 0:
                bb_classes_totals[obj["cls"]] = bb_classes_totals[obj["cls"]] - 1
            else:
                sorted_by_conf.remove(obj)
        
    return sorted_by_conf

def format_outputs(results, img_width, img_height, th):
    xmin = 9999
    xmax = 0
    ymin = 9999
    ymax = 0

    # Set the buuldings we look at
    building_of_interest_burntbase = [
        "eagle",
        "inferno",
        "air_def",
        "scatter",
        "th",
        "wiz_tower",
        "xbow",
    ]

    # Use this to determine the TH level
    th_level = None
    th_level_dict = {
        "eagle": 0,
        "inferno": 0,
        "air_def": 0,
        "scatter": 0,
        "th": 0,
        "wiz_tower": 0,
        "xbow": 0,
        "champ": 0,
        "warden": 0,
    }

    # Loop through the results and increment the th_level_dict
    # Also set the min/max
    for obj in results:
        if obj["cls"] in th_level_dict:
            th_level_dict[obj["cls"]] += 1

        if obj["cls"] in building_of_interest_burntbase:
            if xmin > (obj["xmin"] + obj["xmax"]) / 2:
                xmin = (obj["xmin"] + obj["xmax"]) / 2
            if xmax < (obj["xmin"] + obj["xmax"]) / 2:
                xmax = (obj["xmin"] + obj["xmax"]) / 2
            if ymin > (obj["ymin"] + obj["ymax"]) / 2:
                ymin = (obj["ymin"] + obj["ymax"]) / 2
            if ymax < (obj["ymin"] + obj["ymax"]) / 2:
                ymax = (obj["ymin"] + obj["ymax"]) / 2

    # Set the TH level
    if th == 15:
        th_level = 15
    elif th == 14:
        th_level = 14
    elif th == 13:
        th_level = 13
    else:
        if th_level_dict["scatter"] >= 1 or th_level_dict["champ"] == 1:
            th_level = 13
        elif th_level_dict["inferno"] >= 3:
            th_level = 12
        elif th_level_dict["eagle"] >= 1 or th_level_dict["warden"] >= 1:
            th_level = 11
        else:
            th_level = 10

    # Define the default return dict
    pos_dict = {}
    pos_dict["air_def"] = {}
    pos_dict["air_def"]["positions"] = []
    pos_dict["eagle"] = {}
    pos_dict["eagle"]["positions"] = []
    pos_dict["inferno"] = {}
    pos_dict["inferno"]["positions"] = []
    pos_dict["scatter"] = {}
    pos_dict["scatter"]["positions"] = []
    pos_dict["th"] = {}
    pos_dict["th"]["positions"] = []
    pos_dict["wiz_tower"] = {}
    pos_dict["wiz_tower"]["positions"] = []
    pos_dict["xbow"] = {}
    pos_dict["xbow"]["positions"] = []

    base_width = xmax - xmin
    base_height = ymax - ymin

    rel_base_width = base_width / img_width
    rel_base_height = base_height / img_height
    rel_starting_x = xmin / img_width
    rel_starting_y = ymin / img_height

    # Loop through the results and add the positions to the pos_dict
    for obj in results:
        if obj["cls"] in pos_dict:
            pos_dict[obj["cls"]]["positions"].append(
                {
                    "xPercent": round(
                        ((obj["xmin"] + obj["xmax"]) / 2 - xmin) / base_width, 14
                    ),
                    "yPercent": round(
                        ((obj["ymin"] + obj["ymax"]) / 2 - ymin) / base_height, 14
                    ),
                    "xMinPercent": round((obj["xmin"] - xmin) / base_width, 14),
                    "yMinPercent": round((obj["ymin"] - ymin) / base_height, 14),
                    "xMaxPercent": round((obj["xmax"] - xmin) / base_width, 14),
                    "yMaxPercent": round((obj["ymax"] - ymin) / base_height, 14),
                }
            )

    # Final response dict
    json_dict = {
        "th_level": th_level,
        "coord": {
            "ad": {
                "positions": pos_dict["air_def"]["positions"],
                "buildingName": "Air Defense",
            },
            "ea": {
                "positions": pos_dict["eagle"]["positions"],
                "buildingName": "Eagle Artillery",
            },
            "it": {
                "positions": pos_dict["inferno"]["positions"],
                "buildingName": "Inferno Tower",
            },
            "ss": {
                "positions": pos_dict["scatter"]["positions"],
                "buildingName": "Scattershot",
            },
            "th": {
                "positions": pos_dict["th"]["positions"],
                "buildingName": "Townhall",
            },
            "wt": {
                "positions": pos_dict["wiz_tower"]["positions"],
                "buildingName": "Wizard Tower",
            },
            "xb": {"positions": pos_dict["xbow"]["positions"], "buildingName": "X-Bow"},
        },
        "matrix": {
            "size": {"width": rel_base_width, "height": rel_base_height},
            "startingPosition": {"x": rel_starting_x, "y": rel_starting_y},
        },
    }

    return json_dict
