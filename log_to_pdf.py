
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

def create_pdf(log_filename, output_filename):
    def parse_log(log_filename):
        with open(log_filename, 'r') as f:
            lines = f.readlines()
        return lines

    log_data = parse_log(log_filename)
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    for line in log_data:
        para = Paragraph(line, style=styles["BodyText"])
        flowables.append(para)

    doc.build(flowables)





# testing code
# log_data = parse_log("../logs/data_cleaning_test_2023_10_02_21_20_09.log")
# create_pdf(log_data, "../logs/output_report.pdf")

