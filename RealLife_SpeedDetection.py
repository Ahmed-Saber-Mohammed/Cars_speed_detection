import cv2
import dlib
import time
from datetime import datetime
import os
import numpy as np
from flask import Flask, jsonify, request
import base64
import threading

app = Flask(__name__)

# CLASSIFIER FOR DETECTING CARS--------------------------------------------------
carCascade = cv2.CascadeClassifier("files/HaarCascadeClassifier.xml")

# TAKE VIDEO---------------------------------------------------------------------
video = cv2.VideoCapture(0)  # Or replace with a video file path

WIDTH = 1280  # WIDTH OF VIDEO FRAME
HEIGHT = 720  # HEIGHT OF VIDEO FRAME
cropBegin = 240  # CROP VIDEO FRAME FROM THIS POINT
mark1 = 120  # MARK TO START TIMER
mark2 = 360  # MARK TO END TIMER
markGap = 15  # DISTANCE IN METRES BETWEEN THE MARKERS
fpsFactor = 3  # TO COMPENSATE FOR SLOW PROCESSING
speedLimit = 30  # SPEEDLIMIT
startTracker = {}  # STORE STARTING TIME OF CARS
endTracker = {}  # STORE ENDING TIME OF CARS

# MAKE DIRCETORY TO STORE OVER-SPEEDING CAR IMAGES
if not os.path.exists("overspeeding/cars/"):
    os.makedirs("overspeeding/cars/")

print(f"Speed Limit Set at {speedLimit} Kmph")


def blackout(image):
    xBlack = 360
    yBlack = 300
    triangle_cnt = np.array([[0, 0], [xBlack, 0], [0, yBlack]])
    triangle_cnt2 = np.array([[WIDTH, 0], [WIDTH - xBlack, 0], [WIDTH, yBlack]])
    cv2.drawContours(image, [triangle_cnt], 0, (0, 0, 0), -1)
    cv2.drawContours(image, [triangle_cnt2], 0, (0, 0, 0), -1)

    return image


# FUCTION TO SAVE CAR IMAGE, DATE, TIME, SPEED ----------------------------------
def saveCar(speed, image):
    now = datetime.today().now()
    nameCurTime = now.strftime("%d-%m-%Y-%H-%M-%S-%f")

    link = "overspeeding/cars/" + nameCurTime + ".jpeg"
    cv2.imwrite(link, image)


# FUNCTION TO CALCULATE SPEED----------------------------------------------------
def estimateSpeed(carID):
    timeDiff = endTracker[carID] - startTracker[carID]
    speed = round(markGap / timeDiff * fpsFactor * 3.6, 2)
    return speed


# FUNCTION TO TRACK CARS---------------------------------------------------------
def process_frame(image, carTracker, currentCarID, startTracker, endTracker):
    rectangleColor = (255, 0, 0)
    frameTime = time.time()
    resultImage = blackout(image)
    cv2.line(resultImage, (0, mark1), (1280, mark1), (0, 0, 255), 2)
    cv2.line(resultImage, (0, mark2), (1280, mark2), (0, 0, 255), 2)

    # DELETE CARIDs NOT IN FRAME---------------------------------------------
    carIDtoDelete = []

    for carID in carTracker.keys():
        trackingQuality = carTracker[carID].update(image)

        if trackingQuality < 7:
            carIDtoDelete.append(carID)

    for carID in carIDtoDelete:
        carTracker.pop(carID, None)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cars = carCascade.detectMultiScale(
        gray, 1.1, 13, 18, (24, 24)
    )  # DETECT CARS IN FRAME

    for _x, _y, _w, _h in cars:
        # GET POSITION OF A CAR
        x = int(_x)
        y = int(_y)
        w = int(_w)
        h = int(_h)

        xbar = x + 0.5 * w
        ybar = y + 0.5 * h

        matchCarID = None

        # IF CENTROID OF CURRENT CAR NEAR THE CENTROID OF ANOTHER CAR IN
        # PREVIOUS FRAME THEN THEY ARE THE SAME
        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            tx = int(trackedPosition.left())
            ty = int(trackedPosition.top())
            tw = int(trackedPosition.width())
            th = int(trackedPosition.height())

            txbar = tx + 0.5 * tw
            tybar = ty + 0.5 * th

            if (
                (tx <= xbar <= (tx + tw))
                and (ty <= ybar <= (ty + th))
                and (x <= txbar <= (x + w))
                and (y <= tybar <= (y + h))
            ):
                matchCarID = carID

        if matchCarID is None:
            tracker = dlib.correlation_tracker()
            tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

            carTracker[currentCarID] = tracker
            currentCarID += 1

    for carID in carTracker.keys():
        trackedPosition = carTracker[carID].get_position()

        tx = int(trackedPosition.left())
        ty = int(trackedPosition.top())
        tw = int(trackedPosition.width())
        th = int(trackedPosition.height())

        # PUT BOUNDING BOXES-------------------------------------------------
        cv2.rectangle(
            resultImage, (tx, ty), (tx + tw, ty + th), rectangleColor, 2
        )
        cv2.putText(
            resultImage,
            str(carID),
            (tx, ty - 5),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 0, 0),
            1,
        )

        # ESTIMATE SPEED-----------------------------------------------------
        if (
            carID not in startTracker
            and mark2 > ty + th > mark1
            and ty < mark1
        ):
            startTracker[carID] = frameTime

        elif (
            carID in startTracker
            and carID not in endTracker
            and mark2 < ty + th
        ):
            endTracker[carID] = frameTime
            speed = estimateSpeed(carID)
            if speed > speedLimit:
                print("CAR-ID : {} : {} kmph - OVERSPEED".format(carID, speed))
                saveCar(speed, image[ty : ty + th, tx : tx + tw])
                cv2.rectangle(
                    resultImage, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 2
                )
                cv2.putText(
                    resultImage,
                    str(carID),
                    (tx, ty - 5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    resultImage,
                    "OverSpeed",
                    (tx + 50, ty - 5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 0, 255),
                    1,
                )

            else:
                print("CAR-ID : {} : {} kmph".format(carID, speed))
                cv2.rectangle(
                    resultImage, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2
                )
                cv2.putText(
                    resultImage,
                    str(carID),
                    (tx, ty - 5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )

    ret, jpeg = cv2.imencode(".jpg", resultImage)
    frame_base64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

    return (
        frame_base64,
        carTracker,
        currentCarID,
        startTracker,
        endTracker,
    )




# process_video_stream endpoint: Now expects a list of base64 encoded frames in the "frames" field of the JSON payload. This allows you to send a sequence of frames for processing.
# Car tracking initialization: carTracker, currentCarID, startTracker, and endTracker are initialized outside the loop that processes the frames. This is CRUCIAL! They need to persist across multiple frames to track the same cars over time.
# process_frame function: This function now takes carTracker, currentCarID, startTracker, and endTracker as input, and returns the updated values. This allows the tracking information to be passed from one frame to the next. The image processing also calls blackout now to hide some parts of the image.
# Looping through frames: The endpoint iterates through the list of frames, decoding each one, processing it, and appending the processed frame to a list.
# JSON response: Returns a JSON response containing a list of base64 encoded processed frames.
@app.route("/process_video_stream", methods=["POST"])
def process_video_stream():
    data = request.get_json()
    frames_data = data["frames"]  # List of base64 encoded frames

    # Initialize car tracker and car ID (only once for the entire stream)
    carTracker = {}
    currentCarID = 0
    startTracker = {}
    endTracker = {}

    processed_frames = []

    for frame_data in frames_data:
        # Decode base64 string to numpy array
        frame_bytes = base64.b64decode(frame_data)
        frame_np = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_np, cv2.COLOR_BGR2RGB)

        (
            processed_frame,
            carTracker,
            currentCarID,
            startTracker,
            endTracker,
        ) = process_frame(
            frame, carTracker, currentCarID, startTracker, endTracker
        )  # Process the frame

        processed_frames.append(processed_frame)

    return jsonify({"frames": processed_frames})  # Return processed frames


if __name__ == "__main__":
    app.run(debug=True)
