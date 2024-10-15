colors = ["#8B0000", "#e93e3a", "#ed683c", "#f3903f", "#fdc70c", "#fff33b"]
values = [50, 40, 30, 20, 10, 0]

def value_to_color(value):
    for i in range(6):
        if value > values[i]:
            return colors[i]
        

print(value_to_color(3))