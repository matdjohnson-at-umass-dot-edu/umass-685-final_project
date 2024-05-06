

class Utils:

    def __init__(self):
        pass

    @staticmethod
    def load_python_object(object_path: str, object_attribute: str):
        path_segments = object_path.split('.')
        module = __import__(object_path)
        for segment in path_segments[1:]:
            module = getattr(module, segment)
        return getattr(module, object_attribute)
