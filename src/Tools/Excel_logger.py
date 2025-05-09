class createExcelFile:
    def __init__(self, filename = "reference.xlsx"):
        # Creates workbook and worksheet to modify
        self.workbook = openpyxl.Workbook()
        self.worksheet = self.workbook.active
        self.filename = filename
        self.workbook.save(filename)

        # Initialise variable to store data
        self.dataLine = []

    def extend_data(self, data):
        # adds to list self.dataLine
        data = change_to_list(data)
        self.dataLine.extend(data)

    def export_to_excel(self):
        # appends dataLine to sheet and saves file
        self.workbook = load_workbook(self.filename)
        self.worksheet.append(self.dataLine)       
        self.workbook.save(self.filename)
        self.workbook.close
        self.dataLine = []