
from ultralytics import YOLO
import cv2
import time
import threading
class Camera:
    """_summary_
    python=3.10.16
    Klasa do wykrywania obiektow na kamerze
     DetectObjects oraz DrawBoxes mozna wykorzystaca odzielnie, nie trzeba uzywac run
    """

    def __init__(self, model="yolov8n.pt",conf=0.5,selected_classes=0,image_size=640,cam_idx=0):
        """_summary_
        Args:
            model (str, optional): wybor modelu dostepnego w ultralytics. Defaults to "yolov8n.pt".
            conf (float, optional): pewnosc z jaka model dokonuje detekcji obiektu. Defaults to 0.5.
            selected_classes (int, None, list[int] optional): wybór jakie klasy ma wykrywac model. Defaults to 0.
            image_size (int, optional): rozmiar obrazu. Defaults to 640.
            cam_idx (int, optional): indeks kamery. Defaults to 0.
        """
        self.model= YOLO(model)
        self.cam = cv2.VideoCapture(cam_idx)
        self.conf = conf
        self.selected_classes = selected_classes
        self.image_size = image_size
        
    def DetectObjects(self,frame,conf=None,selected_classes=None,image_size=None):
        """_summary_

        Args:
            frame (_type_): klatka z kamery
            conf (float,None optional): jesli nie jest podane to bierze z konstruktora. Defaults to None.
            selected_classes (int,float,list[int], optional): jesli nie jest podane to bierze z konstruktora. Defaults to None.
            image_size (int, optional): jesli nie jest podane to bierze z konstruktora. Defaults to None.

        Returns:
            list[dict]: zwraca liste slownikow zawierjacych informacje o wykrytych obiektach
            class_name (str): nazwa klasy
            confidence (float): pewnosc z jaka model wykryl obiekt
            first_corner (tuple): pierwszy róg prostokata
            second_corner (tuple): drugi róg prostokata
        """
        if conf is None:
            conf = self.conf
            
        if selected_classes is None:
            selected_classes = self.selected_classes
        
        if image_size is None:
            image_size = self.image_size
            
        results = []
        predictions = self.model.predict(frame, conf=conf, classes=selected_classes, imgsz=image_size, device="cpu")
        prediction = predictions[0]
        for box in prediction.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            first_corner = (int(x1), int(y1))
            second_corner = (int(x2), int(y2))
            class_name = self.model.names[int(cls)]
            confidence = round(conf,2)
            results.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "first_corner": first_corner,
                    "second_corner": second_corner
                }
            )
        return results
    
    def DrawBoxes(self,frame,results):
        """_summary_

        Args:
            frame (_type_): klatka z kamery
            results (list[dict]): lista slownikow zawierjacych informacje o wykrytych obiektach

        Returns:
            _type_: klatka z narysowanymi prostokatami
        """
        for result in results:
            first_corner = result["first_corner"]
            second_corner = result["second_corner"]
            class_name = result["class_name"]
            confidence = result["confidence"]
            color = (0, 255, 0)
            cv2.rectangle(frame, first_corner, second_corner, color, 2)
            cv2.putText(frame, f"{class_name} {confidence}", (first_corner[0], first_corner[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


    
    def run(self):
        """_summary_
        uruchamia kamere i wykrywa obiekty
        "q" aby zakonczyc
        """
        while True:
            start = time.time()
            ret, frame = self.cam.read()
            if not ret:
                print("Failed to grab frame")
                break
            results = self.DetectObjects(frame)
            frame = self.DrawBoxes(frame, results)
            stop = time.time()-start
            fps = int(1/stop)
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Camera", frame)
            
            if cv2.waitKey(1) == ord('q'):
                self.cam.release()
                cv2.destroyAllWindows()
                break



    

# %%

def test_no_video_output(delay_=30,model_="yolov8n.pt",conf_=0.5,selected_classes_=0,image_size_=640,cam_idx_=0):
    """_summary_
    
    Args:
        delay_ (int, optional): _description_. Defaults to 30.
        model_ (str, optional): _description_. Defaults to "yolov8n.pt".
        conf_ (float, optional): _description_. Defaults to 0.5.
        selected_classes_ (int, optional): _description_. Defaults to 0.
        image_size_ (int, optional): _description_. Defaults to 640.
        cam_idx_ (int, optional): _description_. Defaults to 0.
    """
    fps_list = []
    stop_flag = threading.Event()

    def kill(delay):
        """_summary_

        Args:
            delay (_type_): _description_
        """
        time.sleep(delay)
        stop_flag.set()
        print("end of program")

    threading.Thread(target=kill, args=(delay_,)).start()

    cam = Camera(model=model_,conf=conf_,selected_classes=selected_classes_,image_size=image_size_,cam_idx=cam_idx_)
    while not stop_flag.is_set():
        start = time.time()
        ret, frame = cam.cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        results = cam.DetectObjects(frame)
        for result in results:
                first_corner = result["first_corner"]
                second_corner = result["second_corner"]
                class_name = result["class_name"]
                confidence = result["confidence"]
        stop = time.time()-start
        fps = int(1/stop)
        fps_list.append(fps)
        
        print(f" classname: {class_name} \n confidence: {confidence} \n first_corner: {first_corner} \n second_corner: {second_corner} \n fps: {fps}")
                
    cam.cam.release()
    stop_flag = False
    fps_mean = sum(fps_list)/len(fps_list)
    print(f"fps mean: {fps_mean}")

# %%
test_no_video_output()


