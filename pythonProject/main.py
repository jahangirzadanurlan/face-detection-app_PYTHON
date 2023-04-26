import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk

class FaceDetectionApp:
    def __init__(self, master, window_title):
        self.master = master
        self.master.title(window_title)
        self.cam = cv2.VideoCapture(0)
        _, frame = self.cam.read()
        self.height, self.width, _ = frame.shape
        self.canvas = tk.Canvas(self.master, width=self.width, height=self.height)
        self.canvas.pack()
        self.delay = 15
        self.update()

    def update(self):
        _, frame = self.cam.read()
        frame = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.master.after(self.delay, self.update)

    def close(self):
        self.cam.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root, "Face Detection App")
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
