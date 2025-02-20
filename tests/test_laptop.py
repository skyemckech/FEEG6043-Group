from laptop import LaptopPilot

def test_laptop():
    Laptop = LaptopPilot(True)
    Laptop.infinite_loop
    assert Laptop.robot_ip == '127.0.0.1'