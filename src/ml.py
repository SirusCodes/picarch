import PIL
import numpy
import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)

def encode(image_uri: str) -> list[numpy.float32]:
    image = PIL.Image.open(image_uri)
    image = numpy.array(image)
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    faces = app.get(rgb_small_frame)

    if faces is None:
        return []

    return [face.embedding for face in faces]

