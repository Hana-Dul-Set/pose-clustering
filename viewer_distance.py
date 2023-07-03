import cv2
import tkinter
from PIL import Image
from PIL import ImageTk

import utils.image_utils as image_utils
import utils.file_utils as file_utils
import utils.pose_utils as pose_utils


class Cv2ToggleImageLabel(tkinter.Frame):
    def __init__(self, master : tkinter.Misc, cv2images : list) -> None:
        tkinter.Frame.__init__(self, master)

        self.images = []
        for cvimage in cv2images:
            image = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            self.images.append(ImageTk.PhotoImage(image=image))

        self.label = tkinter.Label(self, image=self.images[0])
        self.label.pack(side="top")
    
    def reinit(self, cv2images):
        self.images = []
        for cvimage in cv2images:
            image = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            self.images.append(ImageTk.PhotoImage(image=image))
        self.label.destroy()
        self.label = tkinter.Label(self, image=self.images[0])
        self.label.pack(side="top")

    def get(self):
        return self.label.get()
    
    def setimage(self, index : int):
        self.label.configure(image=self.images[index])
    


keypoints_data = file_utils.read_json('../datas/random/pose_dataset_random.json')
image_DIR = '../datas/random/images/'

window=tkinter.Tk()
window.title("YUN DAE HEE")
window.geometry("1280x720+100+100")
imagesize = (400,400)

image_name1 = "N_671654999113663838680061646740.jpg"
image_name2 = "i_33179.jpg"

def get_data(image_name):
    return next(x for x in keypoints_data if x['name'] == image_name)

def gen_imagepair(image_name, size):
    raw_image = cv2.imread(image_DIR + image_name)
    raw_image = image_utils.resize_with_black_borders(raw_image, size)
    
    keypoints = get_data(image_name)['keypoints'][0]
    pose_image = cv2.imread(image_DIR + image_name)
    left, top, right, bottom = image_utils.predict_image_border_after_resize(pose_image, size)
    pose_image = image_utils.resize_with_black_borders(pose_image, size)
    pose_image = pose_utils.render_keypoints_in_frame(pose_image, keypoints, True, (left, top, right, bottom))

    return (raw_image, pose_image)

def get_distance_and_image(image_name1, image_name2):
    data1 = get_data(image_name1)
    data2 = get_data(image_name2)

    repr1 = pose_utils.keypoints2representation(data1['keypoints'][0], data1['size'])
    repr2 = pose_utils.keypoints2representation(data2['keypoints'][0], data2['size'])

    black_image = image_utils.black_image(imagesize)
    pose_utils.render_representation(black_image, repr1, True, (0,0,255))
    pose_utils.render_representation(black_image, repr2, True, (0,255,0))
    
    distance = pose_utils.distance(repr1['pose'].flatten(), repr2['pose'].flatten())

    return distance, black_image

def on_check():
    global image_label1, image_label2, checkvar1, checkvar2

    image_label1.setimage(checkvar1.get())
    image_label2.setimage(checkvar2.get())

def on_return(event):
    global image_name1, image_name2
    global image_label1, image_label2, distance_image, distance_text, checkvar1, checkvar2

    if image_name1 != textbox1.get():
        image_name1 = textbox1.get()
        raw_image, pose_image = gen_imagepair(image_name1, imagesize)
        image_label1.reinit([raw_image, pose_image])
    elif image_name2 != textbox2.get():
        image_name2 = textbox2.get()
        raw_image, pose_image = gen_imagepair(image_name2, imagesize)
        image_label2.reinit([raw_image, pose_image])
    
    image_label1.setimage(checkvar1.get())
    image_label2.setimage(checkvar2.get())

    distance, black_image = get_distance_and_image(image_name1, image_name2)
    distance_text.delete('1.0', tkinter.END)
    distance_text.insert(tkinter.END, str(distance))
    distance_image.reinit([black_image])

window.bind('<Return>', on_return)

raw_image1, pose_image1 = gen_imagepair(image_name1, imagesize)
raw_image2, pose_image2 = gen_imagepair(image_name2, imagesize)

#image view 1
image_view1 = tkinter.Frame(window, width = 300, height = 400)

checkvar1 = tkinter.IntVar()
checkbox1 = tkinter.Checkbutton(image_view1, text = "Show Keypoints", variable = checkvar1, command = on_check)
checkbox1.pack(side="top")
image_label1 = Cv2ToggleImageLabel(image_view1, [raw_image1, pose_image1])
image_label1.pack(side="top")
textbox1 = tkinter.Entry(image_view1, textvariable=str, font="24")
textbox1.insert(0, image_name1)
textbox1.pack(side="bottom", expand=True, fill='both')

image_view1.pack(side = "left")

#image view 2
image_view2 = tkinter.Frame(window, width = 300, height = 400)

checkvar2 = tkinter.IntVar()
checkbox2 = tkinter.Checkbutton(image_view2, text = "Show Keypoints", variable = checkvar2, command = on_check)
checkbox2.pack(side="top")
image_label2 = Cv2ToggleImageLabel(image_view2, [raw_image2, pose_image2])
image_label2.pack(side="top")
textbox2 = tkinter.Entry(image_view2, textvariable=str, font="24")
textbox2.insert(0, image_name2)
textbox2.pack(side="bottom", expand=True, fill='both')

image_view2.pack(side = "left")

image_label1.setimage(checkvar1.get())
image_label2.setimage(checkvar2.get())

#pose distance  view


distance_view = tkinter.Frame(window, width = 300, height = 400)

distance, black_image = get_distance_and_image(image_name1, image_name2)
distance_image = Cv2ToggleImageLabel(distance_view, [black_image])
distance_image.pack(side="top")

distance_text = tkinter.Text(distance_view, font = "24", height = 50)
distance_text.pack(side="top")
distance_text.insert(tkinter.END, str(distance))
#textbox2 = tkinter.Entry(image_view2, textvariable=str, font="24")
#textbox2.pack(side="bottom", expand=True, fill='both')

distance_view.pack(side = "left")


window.mainloop()