class OuterClass:
    class InnerClass:
        def greet(self):
            return "Hello from InnerClass!"
        
if __name__ == "__main__":
    print("OuterClass available:", hasattr(test, "OuterClass"))