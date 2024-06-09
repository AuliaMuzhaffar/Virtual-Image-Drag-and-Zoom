import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
from tkinter import Tk, filedialog

# Function to open a file dialog for image selection
def select_image():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return file_path

# Function to add a new image to the list
def add_new_image():
    new_image_path = select_image()
    if new_image_path:
        img_type = 'png' if 'png' in new_image_path else 'jpeg'
        listImg.append(DragImg(new_image_path, [50, 50], img_type, 1))

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Resolution width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Initialize the hand detector
detector = HandDetector(detectionCon=0.65)

MAX_ZOOM_LEVEL = 5   # Maximum zoom level
MIN_ZOOM_LEVEL = 0.5 # Minimum zoom level

class DragImg:
    """
    A class representing a draggable image.

    Attributes:
        path (str): The path to the image file.
        posOrigin (tuple): The initial position of the image.
        imgType (str): The type of the image file ('png' or other).
        scaleFactor (float): The scale factor of the image.
        img (numpy.ndarray): The image data.
        size (tuple): The current size of the image.
        original_size (tuple): The original size of the image.
        original_img (numpy.ndarray): The original image data.
        posCurrent (tuple): The current position of the image.
        isResizing (bool): Flag indicating if the image is being resized.
    """

    def __init__(self, path, posOrigin, imgType, scaleFactor):
        self.posOrigin = posOrigin
        self.imgType = imgType
        self.path = path
        self.scaleFactor = scaleFactor

        if self.imgType == 'png':
            self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        else:
            self.img = cv2.imread(self.path)

        self.size = self.img.shape[:2]
        self.original_size = self.size  # Store the original size
        self.original_img = self.img.copy()  # Store the original image
        self.posCurrent = self.posOrigin
        self.isResizing = False

    def update(self, cursor, target_scale_factor=1):
        """
        Update the position and scale of the image based on the cursor position.

        Args:
            cursor (tuple): The current position of the cursor.
            target_scale_factor (float, optional): The target scale factor. Defaults to 1.
        """
        ox, oy = self.posCurrent
        h, w = self.size

        if ox < cursor[0] < ox + w and oy < cursor[1] < oy + h:
            if not self.isResizing:
                self.posCurrent = cursor[0] - w // 2, cursor[1] - h // 2
            else:
                self.size = (cursor[0] - ox, cursor[1] - oy)
        else:
            self.isResizing = False

        # Calculate new scale factor with limits
        new_scale_factor = self.scaleFactor * target_scale_factor
        self.scaleFactor = max(min(new_scale_factor, MAX_ZOOM_LEVEL), MIN_ZOOM_LEVEL)

        # Apply scaling if within limits
        new_size = (int(self.original_size[0] * self.scaleFactor), int(self.original_size[1] * self.scaleFactor))
        if new_size[0] > 10 and new_size[1] > 10:  # Avoid too small
            self.size = new_size
            self.img = cv2.resize(self.original_img, self.size, interpolation=cv2.INTER_AREA)

        # Smooth movement
        self.posOrigin = (
            int(self.posOrigin[0] * 0.9 + self.posCurrent[0] * 0.1),
            int(self.posOrigin[1] * 0.9 + self.posCurrent[1] * 0.1),
        )

# Get the initial image
image_path = select_image()

# Check if an image is selected
if not image_path:
    print("No image selected. Exiting.")
    exit()

# Create an instance of DragImg for the selected image
img_type = 'png' if 'png' in image_path else 'jpeg'
selected_img = DragImg(image_path, [50, 50], img_type, 1)
listImg = [selected_img]

movingMode = False
resizingMode = False

def isThumbIndexFingerUp(hand):
    fingers = detector.fingersUp(hand)
    return fingers == [0, 1, 1, 0, 0]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmlist = hands[0]['lmList']
        cursor = (lmlist[8][0], lmlist[8][1])
        thumb_tip = (lmlist[4][0], lmlist[4][1])  # Corrected to get only x, y coordinates
        index_tip = (lmlist[8][0], lmlist[8][1])  # Corrected to get only x, y coordinates
        length, info, img = detector.findDistance(thumb_tip, index_tip, img)

        if length < 40:
            movingMode = True
            resizingMode = False
        elif 40 <= length < 80:
            resizingMode = True
            movingMode = False
        else:
            movingMode = False
            resizingMode = False
        
        if isThumbIndexFingerUp(hands[0]):
            if selected_img:
                listImg.remove(selected_img)
                selected_img = None

        if detector.fingersUp(hands[0]) == [0, 0, 0, 0, 1] and not movingMode and not resizingMode:
            add_new_image()

        if movingMode or resizingMode:
            for imgObject in listImg:
                imgObject.update(cursor)
                if resizingMode:
                    cv2.rectangle(img, imgObject.posCurrent, cursor, (255, 0, 0), 2)

        # Zoom Logic

        BASE_DISTANCE = 60  # Base distance for normal scale, adjust as needed
        SCALE_SENSITIVITY = 0.001  # Sensitivity of scaling, adjust as needed

        if len(hands) == 1:  # Ensuring only one hand is present for zoom
            hand = hands[0]
            lmList = hand["lmList"]  # List of 21 Landmark points
            thumb_tip = (lmList[4][0], lmList[4][1])
            index_tip = (lmList[8][0], lmList[8][1])
            distance, _, _ = detector.findDistance(thumb_tip, index_tip, img)

            # Calculate scale factor based on distance
            if distance != BASE_DISTANCE:
                scale_factor = 1 + (distance - BASE_DISTANCE) * SCALE_SENSITIVITY
            else:
                scale_factor = 1

            for imgObject in listImg:
                imgObject.update(cursor, scale_factor)


    for imgObject in listImg:
        try:
            h, w = imgObject.size
            ox, oy = imgObject.posOrigin
            if imgObject.imgType == 'png':
                img = cvzone.overlayPNG(img, imgObject.img, [ox, oy])
            else:
                img[oy:oy + h, ox:ox + w] = imgObject.img
        except Exception as e:
            print(f"Error: {e}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
