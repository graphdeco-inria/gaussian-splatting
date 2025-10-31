def plus(x):
    return x + 1


class Test:
    def setup_functions(self):
        self.x_plus = plus
        self.y_plus = plus
        self.z_plus = plus

    def __init__(self):
        self.x = 10
        self.y = 20
        self.z = 30
        self.setup_functions()

    def get_plus_x(self):
        return self.x_plus(self.x)

    def get_plus_y(self):
        return self.y_plus(self.y)

    def get_plus_z(self):
        return self.z_plus(self.z)


t = Test()

print(t.x)
print(t.y)
print(t.z)

print("Changed x: ", t.get_plus_x())
print("Changed y: ", t.get_plus_y())
print("Changed z: ", t.get_plus_z())

t.x = 100
t.y = 200
t.z = 300

print("Changed x: ", t.get_plus_x())
print("Changed y: ", t.get_plus_y())
print("Changed z: ", t.get_plus_z())
