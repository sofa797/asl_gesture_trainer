from services.model_service import CLASS_NAMES

class AppState:
    def __init__(self):
        self.class_names = CLASS_NAMES
        self.current_index = 0
        self.current_target = self.class_names[0]
        self.last_verdict = None
        self.mask_faces = True

    def next_letter(self):
        self.current_index = (self.current_index + 1) % len(self.class_names)
        self.current_target = self.class_names[self.current_index]
        self.last_verdict = None

    def reset(self):
        self.last_verdict = None
