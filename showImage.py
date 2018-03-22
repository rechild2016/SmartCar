import tkinter
import time
from PIL import Image, ImageTk

def button_click_exit_mainloop (event):
    event.widget.quit() # this will cause mainloop to unblock.

def show(images):
    print("in fun")
    root = tkinter.Tk()
    root.bind("<Button>", button_click_exit_mainloop)
    root.geometry('+%d+%d' % (200,300))

    old_label_image = None
    for image1 in images:
        root.geometry('%dx%d' % (image1.size[0]*2,image1.size[1]*2))
        tkpi = ImageTk.PhotoImage(image1)
        label_image = tkinter.Label(root, image=tkpi)
        label_image.place(x=0,y=0,width=image1.size[0],height=image1.size[1])
    
        if old_label_image is not None:
            old_label_image.destroy()
        old_label_image = label_image
        root.update()
        time.sleep(0.1)