from roboflow import Roboflow
rf = Roboflow(api_key="eqKQgy6MdotLIqbauqEf")
project = rf.workspace("skyguard-v4jjz").project("aeroplane-type-object-detection-4w2re")
version = project.version(3)
dataset = version.download("yolov8")
                