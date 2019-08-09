import io
from passporteye import read_mrz
import pytesseract
import numpy as np
from PIL import Image
import cv2 as cv
from datetime import datetime
from timeit import default_timer as timer
from passporteye.mrz.text import MRZ

now = datetime.now()

OCR_OFFSET = 30
MRZ_OFFSET = 6


def to_bytes(image):
    """
    converts image to byte values
    :param image: image
    :return: byte values of the image
    """
    im_bytes = io.BytesIO()
    image.save(im_bytes, format="jpeg")
    return im_bytes.getvalue()


def first_mrz_to_dict(img_mrz):
    """
    attempts to read the mrz of the image using passporteye's read_mrz()
    :param img_mrz: image of mrz
    :return: dictionary of relevant information (dict), type of document (str)
    """
    mrz_dict = None
    try:
        img_mrz = to_bytes(img_mrz)
        img_mrz = read_mrz(img_mrz)
        mrz_dict = img_mrz.to_dict()
        type_mrz = mrz_dict.get("mrz_type")
        mrz_dict = get_data(mrz_dict, type_mrz)
        if not check_mrz(mrz_dict):
            mrz_dict = None
    except AttributeError:
        type_mrz = ""
    return mrz_dict, type_mrz


def get_thresh(im):
    """
    calculates the optimal threshold of the image for use when converting mrz into a binary image
    :param im: image
    :return thresh: optimal threshold for converting the passed image into a binary image (int)
    """
    hist = cv.calcHist([im], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        if q1 == 0 or q2 == 0:
            continue
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        # print("m1, m2, q1, q2", m1, m2, q1, q2)
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # print(thresh)
    return thresh


def get_otsu_threshold(img_path):
    """
    calculates the optimal otsu threshold of the image for use when converting non-mrz area into a binary image
    :param img_path: image path (str)
    :return thresh: optimal otsu threshold (int)
    """
    nparr = np.frombuffer(img_path, np.uint8)
    threshold, discard = cv.threshold(cv.GaussianBlur(cv.imdecode(nparr, 0), (5, 5), 0), 0, 255,
                                      cv.THRESH_BINARY + cv.THRESH_OTSU)
    return int(threshold)


def get_binary(im, val):
    """
    creates a binary image from the passed image for use with pytesseract to extract data
    :param im: image
    :param val: value at which a given pixel is converted to black or white (int)
    :return: binary image
    """
    im = im.convert("L")
    im = np.array(im)
    ret, th1 = cv.threshold(im, val, 255, cv.THRESH_BINARY)
    im = Image.fromarray(th1)
    return im


def check_sum(num):
    """
    checks the validity of the passed IIN number
    :param num: IIN number (str)
    :return: whether the number is valid (bool)
    """
    check_two = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
    control = int(num[11])
    arr = [int(d) for d in num[:-1]]
    for index in range(1, 12):
        arr[index - 1] = index * arr[index - 1]
    rem = sum(arr) % 11
    if rem == 10:
        rem = sum([a * b for a, b in zip(arr, check_two)]) % 11
    return rem == control


def get_data(mrz_dict, type_mrz):
    """
    extracts relevant data from the passed dictionary
    :param mrz_dict: dictionary extracted from mrz (dict)
    :param type_mrz: type of document (str)
    :return: dictionary of relevant information (dict)
    """
    relevant_data = {
        "name": mrz_dict.get('names').split(" ")[0],
        "surname": mrz_dict.get('surname'),
        "country": mrz_dict.get('country'),
        "date_of_birth": mrz_dict.get('date_of_birth'),
        "expiration_date": mrz_dict.get('expiration_date'),
        "number": mrz_dict.get('number'),
        "valid_date_of_birth": mrz_dict.get('valid_date_of_birth'),
        "valid_expiration_date": mrz_dict.get('valid_expiration_date'),
    }
    if type_mrz == 'TD1':
        relevant_data["IIN"] = mrz_dict.get('optional1')[:12]
    elif type_mrz == 'TD2':
        relevant_data["IIN"] = mrz_dict.get('date_of_birth') + mrz_dict.get('optional1')[:6]
    elif type_mrz == 'TD3':
        relevant_data["IIN"] = mrz_dict.get('personal_number')[:12]
    relevant_data["IIN_correct"] = check_sum(relevant_data.get("IIN"))
    return relevant_data


def check_mrz(dictionary, previous=None):
    """
    checks the validity of the information on the mrz extracted by read_mrz or pytesseract.
    If the mrz is read by read_mrz, the function checks for "K" chars at the end of the name. If found,
    the program proceeds to validate the "K" chars using pytesseract.
    :param dictionary: dictionary of relevant values extracted from mrz on the current iteration (dict)
    :param previous: dictionary of relevant values extracted from mrz on the previous iteration (dict)
    :return: whether the information is valid (bool)
    """
    if previous is None:
        return dictionary.get("valid_date_of_birth") and \
               dictionary.get("valid_expiration_date") and \
               dictionary.get("IIN_correct") and \
               dictionary.get("name")[-1:] != "K" and \
               dictionary.get("surname")[-1:] != "K"
    else:
        if previous == dictionary:
            return dictionary.get("valid_date_of_birth") and \
                   dictionary.get("valid_expiration_date") and \
                   dictionary.get("IIN_correct")


def get_date(text_list, type_mrz):
    """Retrieves date of issue from the passed list. If passed list was extracted from a passport,
    the function skips the first date found in the list (as it is the birth date)
    and checks the validity of the second date found.
    :param text_list: list of words extracted by pytesseract (list of str)
    :param type_mrz: type of document (str)
    :return: date of issue (str)
    """
    correct_date = None
    first_date = True
    length_list = len(text_list)
    current_date = int(now.strftime("%Y") + now.strftime("%m") + now.strftime("%d"))
    for index in range(length_list):
        test_date = text_list[index]
        length_test_date = len(test_date)
        if length_test_date == 10:
            test_date = "".join(test_date.replace(",", ".").split("."))
            year = test_date[4:]
            month = test_date[2:4]
            day = test_date[:2]
            test_date = year + month + day
            if test_date.isdigit():
                if type_mrz == "TD3" and first_date:
                    first_date = False
                    continue
                if int(test_date) < current_date:
                    correct_date = test_date
                    break
    return correct_date


def get_issuer(text_list):
    """Retrieves the issuer name form the passed list of strings extracted by pytesseract.
    :param text_list: list of words extracted by pytesseract (list of str)
    :return: name of issuer (str)
    """
    issuer = False
    if ("МВД" in text_list) or ("11М" in text_list) or ("РЕСПУБЛИКАСЫ" in text_list) or ("ПМ" in text_list):
        issuer = "МВД Республики Казахстан"
    elif ("МИНИСТЕРСТВО" in text_list) or ("ЮСТИЦИИ" in text_list):
        issuer = "Министерство Юстиции Республики Казахстан"
    elif "INTERNAL" in text_list:
        issuer = "Ministry of Internal Affairs"
    elif "FOREIGN" in text_list:
        issuer = "Ministry of Foreign Affairs"
    # elif "ӘДЛЕТ" in text_list or "ӘДІЛЕТ" in text_list:
    elif "ЭД`ЛЕТ" in text_list or "ЭД!ЛЕТ" in text_list:
        issuer = "ӘДІЛЕТ МИНИСТРЛІГІ"
    return issuer


def photo_extract(image, type_mrz):
    """Extracts the photo of the individual from the ID.
    :param image: image
    :param type_mrz: type of document (str)
    :return: image of the extracted photo
    """
    photo = Image.open(image)
    photo = photo.resize((2000, 1200))
    if type_mrz == "TD1":
        photo = photo.crop((120, 330, 600, 930))
        # photo.show()
    elif type_mrz == "TD2":
        photo = photo.crop((160, 240, 660, 930))
        # photo.show()
    elif type_mrz == "TD3":
        photo = photo.crop((80, 260, 540, 890))
        # photo.show()
    return photo


def crop_image_mrz(img_front, img_back, type_mrz):
    """Keeps only the mrz from the passed image. Determines where to crop based on document type.

    :param img_front: image of the front of the ID (image)
    :param img_back: image of the back of the ID (image)
    :param type_mrz: type of file (str)
    :return: cropped image of mrz (image)
    """
    width_front, height_front = img_front.size
    width_back, height_back = img_back.size
    img_mrz = None
    if type_mrz == "TD1":
        img_mrz = img_back.crop((0, 0.6 * height_back, width_back, height_back))
    elif type_mrz == "TD2_3":
        img_mrz = img_front.crop((0, 0.79 * height_front, width_front, height_front))

    return img_mrz


def crop_image_ocr(img_front, img_back, type_mrz):
    """Keeps only the portion of the image that will be read by pytesseract from the passed image.
    Determines where to crop based on document type.

    :param img_front: image of the front of the ID (image)
    :param img_back: image of the back of the ID (image)
    :param type_mrz: type of file (str)
    :return: cropped image of the section of the image to be extracted by pytesseract (image)
    """
    width_front, height_front = img_front.size
    width_back, height_back = img_back.size
    img_ocr = None
    if type_mrz == "TD1":
        img_ocr = img_back.crop((0, 0, width_back, height_back * 0.6))
    elif type_mrz == "TD2":
        img_ocr = img_back.crop((0, 0, width_back, 0.79 * height_back))
    elif type_mrz == "TD3":
        img_ocr = img_front.crop((0, 0, width_front, 0.79 * height_front))
    return img_ocr


def print_all(correct_dict):
    """Prints all of the correct values found by the program.

    :param correct_dict: dictionary of valid values extracted from the ID (dict)
    :return: None
    """
    print("****************************")
    print("Surname: ", correct_dict.get("surname"))
    print("Name: ", correct_dict.get("name"))
    print("Country: ", correct_dict.get("country"))
    print("IIN: ", correct_dict.get("IIN"))
    print("IIN Correct?: ", correct_dict.get("IIN_correct"))
    print("Number: ", correct_dict.get("number"))
    print("Date of birth: ", correct_dict.get("date_of_birth"))
    print("Issue date: ", correct_dict.get("issue_date"))
    print("Expiry date: ", correct_dict.get("expiration_date"))
    print("Issued by: ", correct_dict.get("issuer"))
    print("****************************")


########################################################################################################################
def scan_id(img_front_path, img_back_path):
    """
    Main function. Assumes the ID is of type "TD1" crops the two images accordingly. Attempts to extract data from mrz
    using pypassport's read_mrz. If successful, moves onto extracting information from non-mrz image. If unsuccessful,
    enters for loop and attempts to extract mrz information using pytesseract. If upon two iterations no mrz is found,
    changes type of document to ID2_3 (same location of mrz for both), crops accordingly and attempts to read mrz from
    new images. If no mrz is found after two iterations the images are invalid, quits.

    Functions get_thresh() and get_otsu_thresh() calculate the optimal thresholds of mrz and non-mrz images respectively
    for use with image conversion into binary. OFFSETS are used to calibrate the calculated thresholds.

    For loop exists in case pytesseract is not able to effectively parse the binary image, and thus the threshold value
    for use with get_binary() is increased. For mrz, if two iterations in a row provide valid data and are equal to each
    other then the correct_mrz has been found. For correct_issue_date and correct_issuer one valid iteration suffices.
    If either valid mrz or valid issuer or valid issue date is not found after 20 iterations, returns None.

    :param img_front_path: path to the front side of the image
    :param img_back_path: path to the reverse side of the image
    :return: dictionary of correct values extracted from the image or None if valid values were not found (dict),
             image of the extracted photo (image)
    """
    start_time = timer()

    previous_mrz = {}
    correct_issue_date = None
    correct_issuer = None
    mrz_present = True
    mrz_type = "TD1"
    begin_mrz = 0

    image_front = Image.open(img_front_path)
    image_back = Image.open(img_back_path)
    image_ocr = crop_image_ocr(image_front, image_back, mrz_type)
    image_mrz = crop_image_mrz(image_front, image_back, mrz_type)

    correct_mrz, mrz_type = first_mrz_to_dict(image_mrz)

    if not correct_mrz:
        begin_mrz = get_thresh(np.array(image_mrz)) - MRZ_OFFSET
    begin_ocr = get_otsu_threshold(to_bytes(image_ocr)) - OCR_OFFSET

    correction = 0
    for i in range(0, 20):
        if i == 10:
            correction += 20
        if not (correct_mrz and correct_issuer and correct_issue_date):
            if not correct_mrz:
                binary_mrz = get_binary(image_mrz, begin_mrz + i - correction)
                ocr_text = pytesseract.image_to_string(binary_mrz, lang='eng')
                if "<" in ocr_text:
                    mrz = MRZ.from_ocr(ocr_text)
                    mrz_data = mrz.to_dict()
                    mrz_type = mrz_data.get("mrz_type")
                    current_mrz = get_data(mrz_data, mrz_type)
                    if check_mrz(current_mrz, previous_mrz):
                        correct_mrz = current_mrz
                    else:
                        previous_mrz = current_mrz
                else:
                    if not mrz_present:
                        try:
                            mrz_type = "TD2_3"
                            image_mrz = crop_image_mrz(image_front, image_back, mrz_type)
                            correct_mrz, mrz_type = first_mrz_to_dict(image_mrz)
                            if not correct_mrz:
                                begin_mrz = get_otsu_threshold(to_bytes(image_mrz)) - MRZ_OFFSET
                            image_ocr = crop_image_ocr(image_front, image_back, mrz_type)
                            begin_ocr = get_otsu_threshold(to_bytes(image_ocr)) - OCR_OFFSET
                            correction = i
                        except AttributeError:
                            break
                    else:
                        mrz_present = False
                    continue

            if not (correct_issue_date and correct_issuer):
                binary_ocr = get_binary(image_ocr, i + begin_ocr - correction)
                ocr_text = pytesseract.image_to_string(binary_ocr, lang='rus+eng').split()
                if not correct_issuer:
                    correct_issuer = get_issuer(ocr_text)
                if not correct_issue_date:
                    correct_issue_date = get_date(ocr_text, mrz_type)
                    # try:
                    #     correct_issue_date = get_date(ocr_text, mrz_type)
                    # except ValueError:
                    #     ocr_text = pytesseract.image_to_string(binary_ocr, lang='eng+rus').split()
        else:
            break

    if not correct_mrz or not correct_issuer or not correct_issue_date:
        print()
        print("***RETAKE IMAGE***")
        return None, None

    else:
        photo = photo_extract(img_front_path, mrz_type)
        correct_mrz["issue_date"] = correct_issue_date[2:]
        correct_mrz["issuer"] = correct_issuer
        print_all(correct_mrz)
        stop_time = timer()
        print("Duration: ", stop_time - start_time)
        return correct_mrz, photo


scan_id("app_photos/app_sasha_front.jpeg", "app_photos/app_adil_back.jpeg")
scan_id("app_photos/app_sasha_front.jpeg", "app_photos/app_abay_back.jpeg")
scan_id("app_photos/app_sasha_front.jpeg", "app_photos/app_sasha_back.jpeg")
scan_id("app_photos/app_sasha_front.jpeg", "app_photos/app_dima_back.jpeg")
scan_id("app_photos/app_sasha_front.jpeg", "app_photos/app_gulnaz_back.jpeg")
scan_id("app_photos/app_sasha_front.jpeg", "app_photos/app_aset_back.jpeg")
scan_id("app_photos/app_sasha_front.jpeg", "app_photos/app_aldiar_back.jpeg")
scan_id("app_photos/app_sasha_front.jpeg", "app_photos/app_zhanel_back.jpeg")
scan_id("app_photos/passport_marzhan.jpg", "app_photos/blank.jpeg")
scan_id("app_photos/app_sasha_front.jpeg", "app_photos/app_marzhan_good.jpeg")
scan_id("app_photos/app_sasha_front.jpeg", "app_photos/app_marzhan_bad.jpeg")
scan_id("app_photos/app_sasha_front.jpeg", "app_photos/app.jpeg")
scan_id("app_photos/lamin.jpeg", "app_photos/lamin_back.jpeg")
scan_id("app_photos/blank.jpeg", "app_photos/blank.jpeg")
