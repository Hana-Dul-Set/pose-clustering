import tkinter as tk
from PIL import ImageTk, Image

import utils.file_utils as file_utils
CLUSTER_JSON = "../sandbox/datas/pose_cluster_kmeans_result_06261725.json"
DATA_JSON = "../sandbox/datas/filtered_photo_data_0622.json"
IMG_DIR = "../sandbox/datas/all_images/"

class ClusterView:
    def __init__(self, root, image_dir):
        self.root = root
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.main_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', self.configure_canvas)
        self.canvas.bind("<ButtonPress-1>", self.start_scroll)
        self.canvas.bind("<B1-Motion>", self.drag_scroll)

        self.inner_frame = tk.Frame(self.canvas)

        self.start_x = 0
        self.start_y = 0
        self.scrollbar = None

        self.cell_size = (200,200)
        self.image_dir = image_dir
        self.clusters = {}
        self.current_cluster = 0

    def show(self):
        self.inner_frame.pack()
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor=tk.NW)
        
        self.root.mainloop()

    def begin_cluster(self, label):
        self.clusters[label] = []
        self.current_cluster = label
        pass
    def add_image_data(self, image_name, keypoints):
         self.clusters[self.current_cluster].append({'name':image_name, 'keypoints' : keypoints})
    def end_cluster(self):
        for image_data in self.clusters[self.current_cluster]:
            image_path = self.image_dir + image_data['name']
            image = Image.open(image_path)
            resized_image = image.resize(self.cell_size)  # Adjust the size of the displayed images as needed
            tk_image = ImageTk.PhotoImage(resized_image)
            label = tk.Label(self.inner_frame, image=tk_image)
            label.pack(padx=10, pady=10)
        

    def configure_canvas(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def start_scroll(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def drag_scroll(self, event):
        delta_x = event.x - self.start_x
        delta_y = event.y - self.start_y
        self.canvas.xview_scroll(-delta_x, "units")
        self.canvas.yview_scroll(-delta_y, "units")
        self.start_x = event.x
        self.start_y = event.y


# Create the main window
window = tk.Tk()
window.title("Image List with Scroll")
window.geometry("800x600")

clustered_data = file_utils.read_json(CLUSTER_JSON)
pose_data = file_utils.read_json(DATA_JSON)

pose = {}
for record in pose_data:
    pose[record['name']] = record['keypoints'][0]

# Create the ScrollableImageList instance
scrollable_list = ClusterView(window, IMG_DIR)
scrollable_list.begin_cluster("0")

for img_name in clustered_data['groups']["0"]:
    scrollable_list.add_image_data(img_name, pose[img_name])

scrollable_list.end_cluster()

scrollable_list.show()