y = {"red":3, "blue":4, "green":2, "yellow":5}
print({key: value for key, value in sorted(y.items(), key=lambda item: item[1], reverse=True)})