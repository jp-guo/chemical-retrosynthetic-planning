class HeapNode:
    def __init__(self, data, value) -> None:
        self.data = data 
        self.value = value
        
    def __lt__(self, other):
        return self.value < other.value