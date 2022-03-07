import cv2
import ipywidgets as widgets
from IPython.display import display
from tkinter import *
from tkinter import ttk
from matplotlib import pyplot as plt


def template_matching(tmp, pic):
    method = ['cv2.TM_CCOEFF_NORMED']
    image = cv2.imread(pic, 0)
    template = cv2.imread(tmp, 0)
    w, h = template.shape[::-1]

    for meth in method:
        img = image.copy()
        method = eval(meth)

        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, top_left = cv2.minMaxLoc(res)
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img, top_left, bottom_right, 0, 1)

        plt.subplot(121)
        plt.imshow(res, cmap='gray')
        plt.title('Matching Result')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(img, cmap='gray')
        plt.title('Detected Point')
        plt.xticks([])
        plt.yticks([])
        plt.suptitle(meth)
        plt.show()

templates = ['template_eye.jpg', 'template_nose.jpg', 'template_half_face.jpg', 'template_face.jpg']
test_images = ['face_1.jpg', 'face_2.jpg', 'face_3.jpg', 'face_defect_1.jpg', 'face_defect_2.jpg', 'face_defect_3.jpg']


def viola_jones(image_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml',
    )

    line_width = 1
    face_color = (0, 0, 255)
    eyes_color = (0, 255, 0)
    scale_factor = 1.1
    min_neighbors = 6

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, 7)

    faces = face_cascade.detectMultiScale(
        gray, scale_factor, min_neighbors,
    )

    for x, y, w, h in faces:
        img = cv2.rectangle(
            img, (x, y), (x + w, y + h), face_color, line_width,
        )

        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for source, ey, ew, eh in eyes:
            cv2.rectangle(
                roi_color,
                (source, ey),
                (source + ew, ey + eh),
                eyes_color,
                line_width,
            )

    plt.imshow(img)
    cv2.destroyAllWindows()
    plt.show()


def click_button():
    if (method_entry.get() == "Template Matching"):
        template_matching(template_entry.get(), image_entry.get())

    if (method_entry.get() == "Viola Jones"):
        viola_jones(image_entry.get())


root = Tk()
canvas = Canvas(root, width=800, height=400, bg='white')

method_entry = ttk.Combobox(root, state='readonly', values=["Template Matching", "Viola Jones"])
lab = Label(text="method", bg='white')

image_entry = Entry()
image_lab = Label(text="image", bg='white')

template_entry = Entry()
template_lab = Label(text="template", bg='white')

select_button = Button(text="Выбрать", command=click_button)

lab.place(x=5, y=5)
method_entry.place(x=5, y=30)

image_lab.place(x=5, y=75)
image_entry.place(x=5, y=100)

template_lab.place(x=5, y=145)
template_entry.place(x=5, y=170)
select_button.place(x=5, y=195)

canvas.pack()
root.mainloop()