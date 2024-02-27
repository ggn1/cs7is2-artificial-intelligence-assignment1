class MazeState:
    """ 
    State class for a given space in the maze such that
    (x, y) position associated with each action (direction)
    from this position is defined.
    """
    def __init__(self, up, down, left, right):
        self.surroundings = {"↑": up, "→": right, "↓": down, "←":left}

    def __str__(self):
        """
        Surroundings of this maze represented as a string.
        """
        return str(self.surroundings)
    
    def __getitem__(self, key):
        """
        Returns position of new state from surroundings
        based on given key = action that can be taken
        from this state.
        """
        return self.surroundings[key]