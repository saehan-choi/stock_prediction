class CustomNumbers:
    def __init__(self):
        self._numbers = 5
        
    def __getitem__(self, idx):
        return self._numbers[idx]

a = CustomNumbers()

print(a._numbers)

