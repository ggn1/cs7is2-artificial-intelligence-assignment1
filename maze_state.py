class MazeState:
    """State class for a given space in gridworld, with directional attributes pointing to other squares.
    Each directional attribute is a tuple of coordinates (x, y). """

    def __init__(self, up, down, left, right):
        self.surroundings = {'up': up, 'right': right, 'down': down, 'left':left}

    def __str__(self):
        return str(self.surroundings)
    
    def __getitem__(self, key):
        return self.surroundings[key]