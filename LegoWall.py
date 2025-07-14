import sys

class Brick:
  catalog_number: str
  width: int
  height: int
  color: str
  price: float

def lego_wall(w, h):
  # TODO: Implement the function logic
  print(w, h)
  pass

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python script.py <value1> <value2>")
    print("Example: python script.py 9 3")
    sys.exit(1)
  
  width = sys.argv[1]
  height = sys.argv[2]
  lego_wall(int(width), int(height))