import runpy
import cv2
import mysql.connector
import numpy as np
import logging
import time
from ultralytics import YOLO
import pandas as pd

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create file handler and set formatter
log_path = log_path = r'C:\Users\cang8\Desktop\log\file.log'
file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)

# connect db
mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="531642.",
        port="3309",
        database="pasadvanced"
    )

mycursorHdmi = mydb.cursor()
mycursorHdmi1 = mydb.cursor()
mycursorHdmi2 = mydb.cursor()
mycursorSatellite = mydb.cursor()

# function that execute hdmi1 query
def resultHdmi1():
    resultHdmi1 = False
    mycursorHdmi1.execute("select bHdmi1 from pasadvanced.hdmi")
    resultHdmi1 = mycursorHdmi1.fetchone()[0]
    return resultHdmi1

# function that execute hdmi2 query
def resultHdmi2():
    resultHdmi2 = False
    mycursorHdmi2.execute("select bHdmi2 from pasadvanced.hdmi")
    resultHdmi2 = mycursorHdmi2.fetchone()[0]
    return resultHdmi2

# function that execute productid on hdmi
def hdmiProductId():
    hdmiProductId = False
    mycursorHdmi.execute("select u32recProductId from pasadvanced.hdmi")
    hdmiProductId = mycursorHdmi.fetchone()[0]
    return hdmiProductId

# function that execute satellite query
def satellite():
    resultSatellite = False
    mycursorSatellite.execute("select bSatellite from pasadvanced.satellite")
    resultSatellite = mycursorSatellite.fetchone()[0]
    return resultSatellite

# function that execute productid on satellite
def satelliteProductId():
    satelliteProductId = False
    mycursorSatellite.execute("select u32recProductId from pasadvanced.satellite")
    satelliteProductId = mycursorSatellite.fetchone()[0]
    return satelliteProductId

# control camera
cap = cv2.VideoCapture(0)
hasOpened = False
if cap.isOpened() == True:
    logger.info('Camera worked successfully')
else:
    logger.error('Camera does not work')

# logo file
model = YOLO('best.pt')

# control function hdmi1
def hdmi1():
    while True:
        # detect colors red and white
        ret, frame = cap.read()
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_lower = np.array([160, 155, 84], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)

        white_lower = np.array([0, 0, 200], np.uint8)
        white_upper = np.array([100, 0, 255], np.uint8)

        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

        kernel = np.ones((5, 5), "uint8")

        red_mask = cv2.dilate(red_mask, kernel)
        res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

        white_mask = cv2.dilate(white_mask, kernel)
        res_white = cv2.bitwise_and(frame, frame, mask=white_mask)

        # contours
        contoursRed, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contoursWhite, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # detect red color
        def red_color():
            for pic, contour in enumerate(contoursRed):
                areaRed = cv2.contourArea(contour)
                if (areaRed > 300):
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(hsvFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    return "red"

        # detect white color
        def white_color():
            for pic, contour in enumerate(contoursWhite):
                areaRed = cv2.contourArea(contour)
                if (areaRed > 300):
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(hsvFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "White", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                    return "white"

        resultHdmi1ProductId = hdmiProductId()

        # control hdmi1
        if (red_color() == "red" or white_color() == "white") and resultHdmi1() == 1:
            hdmi1query = """update hdmitestresult set sHdmi1Result='basarili' where u32recProductId = %s"""
            tuple1 = (resultHdmi1ProductId)
            mycursorHdmi1.execute(hdmi1query, (tuple1,))
            mydb.commit()
            time.sleep(5)
            logger.info('hdmi1 basarili')
            print("hdmi1 basarili")
        else:
            hdmi1query = """update hdmitestresult set sHdmi1Result='basarisiz' where u32recProductId = %s"""
            tuple1 = (resultHdmi1ProductId)
            mycursorHdmi1.execute(hdmi1query, (tuple1,))
            mydb.commit()
            time.sleep(5)
            logger.info('hdmi1 basarisiz')
            print("hdmi1 basarisiz")

# control function hdmi2
def hdmi2():
    while True:
        # detect colors red and white
        ret, frame = cap.read()
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_lower = np.array([160, 155, 84], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)

        white_lower = np.array([0, 0, 200], np.uint8)
        white_upper = np.array([100, 0, 255], np.uint8)

        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

        kernel = np.ones((5, 5), "uint8")

        red_mask = cv2.dilate(red_mask, kernel)
        res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

        white_mask = cv2.dilate(white_mask, kernel)
        res_white = cv2.bitwise_and(frame, frame, mask=white_mask)

        # contours
        contoursRed, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contoursWhite, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # detect red color
        def red_color():
            for pic, contour in enumerate(contoursRed):
                areaRed = cv2.contourArea(contour)
                if (areaRed > 300):
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(hsvFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    return "red"

        # detect white color
        def white_color():
            for pic, contour in enumerate(contoursWhite):
                areaRed = cv2.contourArea(contour)
                if (areaRed > 300):
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(hsvFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "White", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                    return "white"

        resultHdmi2ProductId = hdmiProductId()

        # control hdmi2
        if (red_color() == "red" or white_color() == "white") and resultHdmi2() == 1:
            hdmi2query = """update hdmitestresult set sHdmi2Result='basarili' where u32recProductId = %s"""
            tuple2 = (resultHdmi2ProductId)
            mycursorHdmi2.execute(hdmi2query, (tuple2,))
            mydb.commit()
            time.sleep(5)
            logger.info('hdmi2 basarili')
            print("hdmi2 basarili")
        else:
            hdmi2query = """update hdmitestresult set sHdmi2Result='basarisiz' where u32recProductId = %s"""
            tuple2 = (resultHdmi2ProductId)
            mycursorHdmi2.execute(hdmi2query, (tuple2,))
            mydb.commit()
            time.sleep(5)
            logger.info('hdmi2 basarisiz')
            print("hdmi2 basarisiz")

# control function logo
def logo():

    ret, frame = cap.read()

    def logoResult():
        results = model.predict(frame)
        a = results[0].boxes.boxes
        px = pd.DataFrame(a).astype("float")
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if cv2.rectangle:
                print("logo")
                return "logo"

    resultSatelliteProductId = satelliteProductId()

    # save logo result
    if logoResult() == "logo" and satellite() == 1:
        satellitequery = """update satelliteresult set satelliteresult='basarili' where u32recProductId = %s"""
        tuple3 = (resultSatelliteProductId)
        mycursorHdmi2.execute(satellitequery, (tuple3,))
        mydb.commit()
        # time.sleep(5)
        logger.info('yayin basarili')
        print("yayin basarili")
    else:
        satellitequery = """update satelliteresult set satelliteresult='basarisiz' where u32recProductId = %s"""
        tuple3 = (resultSatelliteProductId)
        mycursorHdmi2.execute(satellitequery, (tuple3,))
        mydb.commit()
        # time.sleep(5)
        logger.info('yayin basarisiz')
        print("yayin basarisiz")

# control hdmi1
while resultHdmi1() == 1:
    logger.info('hdmi1 takili')
    print("hdmi1 takılı")
    hdmi1()
    time.sleep(5)
    if resultHdmi1() != 1:
        logger.error('hdmi1 takili degil')
        print("hdmi1 takılı değil")
        while True:
            runpy.run_path('main.py', run_name='__main__')

# control hdmi2
while resultHdmi2() == 1:
    logger.info('hdmi2 takili')
    print("hdmi2 takılı")
    hdmi2()
    time.sleep(5)
    if resultHdmi2() != 1:
        logger.info('hdmi2 takili degil')
        print("hdmi2 takılı değil")
        while True:
            runpy.run_path('main.py', run_name='__main__')

# control logo
while satellite() == 1:
    logger.info('yayin basladi')
    print("yayın başladı")
    logo()
    time.sleep(5)
    runpy.run_path('main.py', run_name='__main__')
    if satellite() != 1:
        logger.error('yayin yok')
        print("yayın yok")
        while True:
            runpy.run_path('main.py', run_name='__main__')

# no cables are connected
if resultHdmi1() != 1 and resultHdmi2() != 1 and satellite() != 1:
    logger.error('program calismiyor hicbir kablo bagli degil')
    print("program çalışmıyor, hiçbir kablo bağlı değil")
    time.sleep(5)
    while True:
        runpy.run_path('main.py', run_name='__main__')